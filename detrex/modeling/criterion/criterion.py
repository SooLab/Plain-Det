# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This is the original implementation of SetCriterion which will be deprecated in the next version.

We keep it here because our modified Criterion module is still under test.
"""

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os

from detrex.layers import box_cxcywh_to_xyxy, generalized_box_iou
from detrex.utils import get_world_size, is_dist_avail_and_initialized

def load_class_freq(dataset_name, freq_weight=1.0):
   path = {
       'lvis_v1':'/public/home/zhuyuchen530/projects/detrex/datasets/metadata/lvis_v1_train_cat_info.json',
       'o365':'/public/home/zhuyuchen530/projects/detrex/datasets/metadata/object365_train_cat_info.json'
   }
   cat_info = json.load(open(path[dataset_name], 'r'))
   cat_info = torch.tensor(
       [c['image_count'] for c in sorted(cat_info, key=lambda x: x['id'])])
   freq_weight = cat_info.float() ** freq_weight
   return freq_weight   

def get_fed_loss_inds(gt_classes, num_sample_cats, C, weight=None):
    appeared = torch.unique(gt_classes) # C'
    prob = appeared.new_ones(C + 1).float()
    prob[-1] = 0
    if len(appeared) < num_sample_cats:
        if weight is not None:
            prob[:C] = weight.float().clone()
        prob[appeared] = 0
        more_appeared = torch.multinomial(
            prob, num_sample_cats - len(appeared),
            replacement=False)
        appeared = torch.cat([appeared, more_appeared])
    return appeared


def _load_class_hierarchy():
        hierarchy_weight = None
        # hierarchy_path = '/storage/data/zhuyuchen530/oid/annotations/challenge-2019-label500-hierarchy-list.json'
        hierarchy_path = '/inspurfs/group/yangsb/zhuyuchen/datasets/OID_V4/annotations/bbox_labels_600_hierarchy-list.json'
        # print('Loading', cfg.MODEL.ROI_BOX_HEAD.HIERARCHY_PATH)
        hierarchy_data = json.load(open(hierarchy_path, 'r'))
        parents = {int(k): v for k, v in hierarchy_data['parents'].items()}
        chirlds = {int(k): v for k, v in hierarchy_data['childs'].items()}
        categories = hierarchy_data['categories']
        continousid = sorted([x['id'] for x in categories])
        catid2continous = {x['id']: continousid.index(x['id']) \
            for x in categories}
        C = len(categories)
        is_parents = torch.zeros((C, C), device=torch.device('cuda')).float()
        is_chirlds = torch.zeros((C, C), device=torch.device('cuda')).float()
        for c in categories:
            cat_id = catid2continous[c['id']]
            is_parents[cat_id, [catid2continous[x] for x in parents[c['id']]]] = 1
            is_chirlds[cat_id, [catid2continous[x] for x in chirlds[c['id']]]] = 1
        assert (is_parents * is_chirlds).sum() == 0
        # if cfg.MODEL.ROI_BOX_HEAD.HIERARCHY_POS_PARENTS:
        hierarchy_weight = (1 - is_chirlds, is_parents[:C])
        # else:
        #     hierarchy_weight = 1 - (is_parents + is_chirlds) # (C + 1) x C
    
        return hierarchy_weight
    

def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2, weight = None):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (torch.Tensor): A float tensor of arbitrary shape.
            The predictions for each example.
        targets (torch.Tensor): A float tensor with the same shape as inputs. Stores the binary
            classification label for each element in inputs
            (0 for the negative class and 1 for the positive class).
        num_boxes (int): The number of boxes.
        alpha (float, optional): Weighting factor in range (0, 1) to balance
            positive vs negative examples. Default: 0.25.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
            balance easy vs hard examples. Default: 2.

    Returns:
        torch.Tensor: The computed sigmoid focal loss.
    """
    prob = inputs.sigmoid()
    # print("shape:",prob.shape)
    # print("prob:",prob)
    # print("target:",targets)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    if weight != None:
        assert ce_loss.shape == weight.shape, f"{ce_loss.shape} not equal to {weight.shape}"
        # print(f"ce_loss:{ce_loss.shape};weight:{weight.shape}")
        ce_loss *= weight
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


class SetCriterion(nn.Module):
    """This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        similarity=None,
        losses: List[str] = ["class", "boxes"],
        eos_coef: float = 0.1,
        loss_class_type: str = "focal_loss",
        alpha: float = 0.25,
        gamma: float = 2.0,
        dataset_name : str = "lvis_v1",
    ):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            similarity: when use text_embedding, (num_class, num_class)
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.dataset_name = dataset_name
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.similarity = similarity
        self.losses = losses
        self.alpha = alpha
        self.gamma = gamma
        self.eos_coef = eos_coef
        self.loss_class_type = loss_class_type
        assert loss_class_type in [
            "ce_loss",
            "focal_loss",
            "fed_loss"
            "oid_loss"
        ], "only support ce loss and focal loss for computing classification loss"
        self.register_buffer('fed_loss_weight_lvis_v1', load_class_freq('lvis_v1', freq_weight=0.5))
        self.register_buffer('fed_loss_weight_o365', load_class_freq('o365', freq_weight=0.5))

        hierarchy_weight = _load_class_hierarchy()
        # if self.pos_parents and (hierarchy_weight is not None):
        if hierarchy_weight is not None:
            self.hierarchy_weight = hierarchy_weight[0] # (C + 1) x C
            self.is_parents = hierarchy_weight[1]
        if self.loss_class_type == "ce_loss":
            empty_weight = torch.ones(self.num_classes + 1)
            empty_weight[-1] = eos_coef
            self.register_buffer("empty_weight", empty_weight)

    
    def loss_labels(self, outputs, targets, indices, num_boxes, **kwargs):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]
        # idx -> (batch, source) 预测的类是几号batch的第几个query
        idx = self._get_src_permutation_idx(indices)
        # batch中所有预测类的预测类的GT
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        # (b, num_queries)
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        # 构建真值表（b, numqueries）-> target class
        target_classes[idx] = target_classes_o

        # Computation classification loss
        if self.loss_class_type == "ce_loss":
            loss_class = F.cross_entropy(
                src_logits.transpose(1, 2), target_classes, self.empty_weight
            )
        elif self.loss_class_type == "focal_loss":
            # src_logits: (b, num_queries, num_classes) = (2, 300, 80)
            # target_classes_one_hot = (2, 300, 80)
            target_classes_onehot = torch.zeros(
                [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                dtype=src_logits.dtype,
                layout=src_logits.layout,
                device=src_logits.device,
            )
            target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
            target_classes_onehot = target_classes_onehot[:, :, :-1]
            loss_class = (
                sigmoid_focal_loss(
                    src_logits,
                    target_classes_onehot,
                    num_boxes=num_boxes,
                    alpha=self.alpha,
                    gamma=self.gamma,
                )
                * src_logits.shape[1]
            )
        elif self.loss_class_type == "fed_loss":
            target_classes_onehot = torch.zeros(
                [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                dtype=src_logits.dtype,
                layout=src_logits.layout,
                device=src_logits.device,
            )
            target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
            target_classes_onehot = target_classes_onehot[:, :, :-1]

            if self.dataset_name == "lvis_v1":
                fed_loss_weight = self.fed_loss_weight_lvis_v1
            if self.dataset_name == "o365":
                fed_loss_weight = self.fed_loss_weight_o365
            inds = get_fed_loss_inds(
                gt_classes=target_classes_o,
                num_sample_cats=50,
                weight=fed_loss_weight,
                C=target_classes_onehot.shape[2]
            )            
            loss_class = (
                sigmoid_focal_loss(
                    src_logits[:,:,inds],
                    target_classes_onehot[:,:,inds],
                    num_boxes=num_boxes,
                    alpha=self.alpha,
                    gamma=self.gamma,
                )
                * src_logits.shape[1]
            )
        elif self.loss_class_type == "oid_loss":
            # src_logits: (b, num_queries, num_classes) = (2, 300, 80)
            # target_classes_one_hot = (2, 300, 80)
            target_classes_onehot = torch.zeros(
                [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                dtype=src_logits.dtype,
                layout=src_logits.layout,
                device=src_logits.device,
            )
            target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
            target_classes_onehot = target_classes_onehot[:, :, :-1]
            weight = torch.ones_like(target_classes_onehot)
            weight[idx] = self.hierarchy_weight[target_classes_o]

            target_classes_onehot = torch.bmm(target_classes_onehot, \
                self.is_parents.unsqueeze(0).repeat(target_classes_onehot.shape[0],1,1)) \
                + target_classes_onehot
            loss_class = (
                sigmoid_focal_loss(
                    src_logits,
                    target_classes_onehot,
                    num_boxes=num_boxes,
                    alpha=self.alpha,
                    gamma=self.gamma,
                    weight=weight,
                )
                * src_logits.shape[1]
            )

        # if 'similarity' in kwargs:
        #     assert idx[-1].shape[-1] == target_classes_o.shape[-1], print(f"nums not match {idx.shape[-1]} and {target_classes_o.shape[-1]}")

        #     loss_mse = sigmod_mse_loss(src_logits[idx], self.similarity[target_classes_o])
        #     # print('********************************')
        #     # print('MSE:',loss_mse) 
        #     # print('FOCAL:',loss_class) 
        #     # print('********************************')
        #     loss_class = 0.8*loss_class + 0.2*loss_mse

        losses = {"loss_class": loss_class}

        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes, **kwargs):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(target_boxes),
            )
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "class": self.loss_labels,
            "boxes": self.loss_boxes,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, return_indices=False):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc

             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.

        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        if return_indices:
            indices0_copy = indices
            indices_list = []

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if return_indices:
            indices_list.append(indices0_copy)
            return losses, indices_list

        return losses


    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "loss_class_type: {}".format(self.loss_class_type),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "focal loss alpha: {}".format(self.alpha),
            "focal loss gamma: {}".format(self.gamma),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


