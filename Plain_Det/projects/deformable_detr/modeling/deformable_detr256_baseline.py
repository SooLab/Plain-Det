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

import copy
import math
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

from detrex.layers import MLP, box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from detrex.utils import inverse_sigmoid

from detectron2.modeling import detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances
import numpy as np

class DeformableDETR256B(nn.Module):
    """Implements the Deformable DETR model.

    Code is modified from the `official github repo
    <https://github.com/fundamentalvision/Deformable-DETR>`_.

    More details can be found in the `paper
    <https://arxiv.org/abs/2010.04159>`_ .

    Args:
        backbone (nn.Module): the backbone module.
        position_embedding (nn.Module): the position embedding module.
        neck (nn.Module): the neck module.
        transformer (nn.Module): the transformer module.
        embed_dim (int): the dimension of the embedding.
        num_classes (int): Number of total categories.
        num_queries (int): Number of proposal dynamic anchor boxes in Transformer
        criterion (nn.Module): Criterion for calculating the total losses.
        pixel_mean (List[float]): Pixel mean value for image normalization.
            Default: [123.675, 116.280, 103.530].
        pixel_std (List[float]): Pixel std value for image normalization.
            Default: [58.395, 57.120, 57.375].
        aux_loss (bool): whether to use auxiliary loss. Default: True.
        with_box_refine (bool): whether to use box refinement. Default: False.
        as_two_stage (bool): whether to use two-stage. Default: False.
        select_box_nums_for_evaluation (int): the number of topk candidates
            slected at postprocess for evaluation. Default: 100.

    """

    def __init__(
        self,
        backbone,
        position_embedding,
        neck,
        transformer,
        embed_dim,
        num_classes,
        num_queries,
        criterion,
        pixel_mean,
        pixel_std,
        output_dir = None,
        label_embedding = None,
        aux_loss=True,
        online_sample=False,
        with_box_refine=False,
        as_two_stage=False,
        select_box_nums_for_evaluation=100,
        device="cuda",
    ):
        super().__init__()
        # define backbone and position embedding module
        self.backbone = backbone
        self.position_embedding = position_embedding

        # label embedding
        self.label_embedding = label_embedding

        self.output_dir = output_dir
        self.online_sample = online_sample

        # define neck module
        self.neck = neck

        # define learnable query embedding
        self.num_queries = num_queries
        if not as_two_stage:
            self.query_embedding = nn.Embedding(num_queries, embed_dim * 2)


        self.name2idx = {
                "coco" : 0,
                "lvis_v1" : 1,
                "o365" : 2,
                "oid" : 3,
        }
        ############   LABEL META QUERY INIT  ###########
        num_label = len(label_embedding)
        label_proj_list = []
        label_coff_list = []
        for _ in range(num_label):
            num_basis = label_embedding[_].shape[0]
            label_in_channels = label_embedding[_].shape[1]
            label_proj_list.append(nn.Sequential(
                nn.Linear(label_in_channels, label_in_channels * 2),
                nn.ReLU(),
                nn.Linear(label_in_channels * 2, embed_dim))
            )

            meta_coff = nn.Parameter(torch.ones([num_queries, num_basis]) * np.log(1 / 0.07))
            label_coff_list.append(meta_coff)
        self.shared_label_proj = nn.Sequential(
                nn.Linear(label_in_channels, label_in_channels * 2),
                nn.ReLU(),
                nn.Linear(label_in_channels * 2, embed_dim))
        self.label_proj_list = nn.ModuleList(label_proj_list)
        self.label_coff_list = nn.ParameterList(label_coff_list)
        
        self.meta_init_query = nn.Embedding(num_queries, embed_dim)
        self.ca_text = nn.MultiheadAttention(embed_dim, 8, dropout=0.0)
        self.catext_norm = nn.LayerNorm(embed_dim)
            
        # define transformer module
        self.transformer = transformer

        # define classification head and box head
        self.num_classes = num_classes
        self.temp = nn.Linear(embed_dim,512)
        self.class_embed_0 = nn.Linear(512, num_classes[0], bias=False)
        self.class_embed_1 = nn.Linear(512, num_classes[1], bias=False)
        self.class_embed_2 = nn.Linear(512, num_classes[2], bias=False)
        self.class_embed_3 = nn.Linear(512, num_classes[3], bias=False)
        self.bbox_embed = MLP(embed_dim, embed_dim, 4, 3)

        # where to calculate auxiliary loss in criterion
        self.aux_loss = aux_loss
        self.criterion = criterion

        # define contoller for box refinement and two-stage variants
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage

        # init parameters for heads
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        self.class_embed_0.weight.data = self.label_embedding[0]
        self.class_embed_1.weight.data = self.label_embedding[1]
        self.class_embed_2.weight.data = self.label_embedding[2]
        self.class_embed_3.weight.data = self.label_embedding[3]
        for _, neck_layer in self.neck.named_modules():
            if isinstance(neck_layer, nn.Conv2d):
                nn.init.xavier_uniform_(neck_layer.weight, gain=1)
                nn.init.constant_(neck_layer.bias, 0)

        # If two-stage, the last class_embed and bbox_embed is for region proposal generation
        # Decoder layers share the same heads without box refinement, while use the different
        # heads when box refinement is used.
        num_pred = (
            (transformer.decoder.num_layers + 1) if as_two_stage else transformer.decoder.num_layers
        )
        if with_box_refine:
            self.class_embed_0 = nn.ModuleList(
                [copy.deepcopy(self.class_embed_0) for i in range(num_pred)]
            )
            self.class_embed_1 = nn.ModuleList(
                [copy.deepcopy(self.class_embed_1) for i in range(num_pred)]
            )
            self.class_embed_2 = nn.ModuleList(
                [copy.deepcopy(self.class_embed_2) for i in range(num_pred)]
            )
            self.class_embed_3 = nn.ModuleList(
                [copy.deepcopy(self.class_embed_3) for i in range(num_pred)]
            )
            self.bbox_embed = nn.ModuleList(
                [copy.deepcopy(self.bbox_embed) for i in range(num_pred)]
            )
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed_0 = nn.ModuleList([self.class_embed_0 for _ in range(num_pred)])
            self.class_embed_1 = nn.ModuleList([self.class_embed_1 for _ in range(num_pred)])
            self.class_embed_2 = nn.ModuleList([self.class_embed_2 for _ in range(num_pred)])
            self.class_embed_3 = nn.ModuleList([self.class_embed_3 for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        
        self.class_embed = {
            "coco":self.class_embed_0, 
            "lvis_v1":self.class_embed_1,
            "o365":self.class_embed_2,
            "oid":self.class_embed_3,
        }

        # hack implementation for two-stage. The last class_embed and bbox_embed is for region proposal generation
        if as_two_stage:
            prior_prob = 0.01
            # bias_value = -math.log((1 - prior_prob) / prior_prob)
            # bias_value=0
            self.transformer.decoder.class_embed_0 = nn.Linear(512, num_classes[0])
            self.transformer.decoder.class_embed_1 = nn.Linear(512, num_classes[1])
            self.transformer.decoder.class_embed_2 = nn.Linear(512, num_classes[2])
            self.transformer.decoder.class_embed_3 = nn.Linear(512, num_classes[3])
            # self.transformer.decoder.class_embed_0.bias.data = torch.ones(num_classes[0]) * bias_value
            # self.transformer.decoder.class_embed_1.bias.data = torch.ones(num_classes[1]) * bias_value
            # self.transformer.decoder.class_embed_2.bias.data = torch.ones(num_classes[2]) * bias_value
            # self.transformer.decoder.class_embed_0.weight.data = self.label_embedding[0]
            # self.transformer.decoder.class_embed_1.weight.data = self.label_embedding[1]
            # self.transformer.decoder.class_embed_2.weight.data = self.label_embedding[2]
            self.transformer.decoder.class_embed_0 = nn.ModuleList([copy.deepcopy(self.transformer.decoder.class_embed_0)for i in range(num_pred)])
            self.transformer.decoder.class_embed_1 = nn.ModuleList([copy.deepcopy(self.transformer.decoder.class_embed_1)for i in range(num_pred)])
            self.transformer.decoder.class_embed_2 = nn.ModuleList([copy.deepcopy(self.transformer.decoder.class_embed_2)for i in range(num_pred)])
            self.transformer.decoder.class_embed_3 = nn.ModuleList([copy.deepcopy(self.transformer.decoder.class_embed_3)for i in range(num_pred)])
            self.transformer.decoder.class_embed = {
                "coco":self.transformer.decoder.class_embed_0, 
                "lvis_v1":self.transformer.decoder.class_embed_1,
                "o365":self.transformer.decoder.class_embed_2,
                "oid":self.transformer.decoder.class_embed_3,
            }

            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

        # set topk boxes selected for inference
        self.select_box_nums_for_evaluation = select_box_nums_for_evaluation

        # normalizer for input raw images
        self.device = device
        pixel_mean = torch.Tensor(pixel_mean).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(pixel_std).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        # freeze sth
        for dataset_name in self.class_embed:
            for _,param in self.class_embed[dataset_name].named_parameters():
                param.requires_grad_(True)

        self.losslog = {
            'coco':[],
            'lvis_v1':[],
            'o365':[],
            'oid':[]
        }

        self.count = 0


    def forward(self, batched_inputs, dataset_name="coco"):
        if dataset_name == "coco":
            # print("we choose coco")
            class_num = self.num_classes[0]
            self.criterion.loss_class_type = "focal_loss"
        elif dataset_name == "lvis_v1":
            # print("we choose lvis")
            class_num = self.num_classes[1]
            self.criterion.loss_class_type = "fed_loss"
        elif dataset_name == "o365":
            class_num = self.num_classes[2]
            self.criterion.loss_class_type = "focal_loss"
        elif dataset_name == "oid":
            class_num = self.num_classes[3]
            self.criterion.loss_class_type = "oid_loss"
        else:
            assert 1 == 0, "dataset_name not registered"

        self.criterion.num_classes = class_num

        images = self.preprocess_image(batched_inputs)

        if self.training:
            batch_size, _, H, W = images.tensor.shape
            img_masks = images.tensor.new_ones(batch_size, H, W)
            for img_id in range(batch_size):
                # mask padding regions in batched images
                img_h, img_w = batched_inputs[img_id]["instances"].image_size
                img_masks[img_id, :img_h, :img_w] = 0
        else:
            batch_size, _, H, W = images.tensor.shape
            img_masks = images.tensor.new_zeros(batch_size, H, W)

        # original features
        features = self.backbone(images.tensor)  # output feature dict

        # project backbone features to the reuired dimension of transformer
        # we use multi-scale features in deformable DETR
        multi_level_feats = self.neck(features)
        multi_level_masks = []
        multi_level_position_embeddings = []
        for feat in multi_level_feats:
            multi_level_masks.append(
                F.interpolate(img_masks[None], size=feat.shape[-2:]).to(torch.bool).squeeze(0)
            )
            multi_level_position_embeddings.append(self.position_embedding(multi_level_masks[-1]))

        # initialize object query embeddings
        query_embeds = None
        if not self.as_two_stage:
            query_embeds = self.query_embedding.weight

        
        global_feature = multi_level_feats[-1].mean((2,3))
        bs = global_feature.shape[0]
        global_feature = global_feature / global_feature.norm(dim=-1, keepdim=True)
        iidx = self.name2idx[dataset_name]
        # iidx = self.name2idx['o365']
        meta_label = self.label_proj_list[iidx](self.label_embedding[iidx])
        meta_label_norm = meta_label / meta_label.norm(dim=-1, keepdim=True)
        global_coff = 100 * global_feature @ meta_label_norm.t()
        init_query_feature = self.meta_init_query.weight[None].repeat(bs, 1, 1)
        
        query_embeds = self.ca_text(
            init_query_feature.transpose(0,1),
            meta_label[None].repeat(bs, 1, 1).transpose(0,1),
            meta_label[None].repeat(bs, 1, 1).transpose(0,1),
            attn_mask = global_coff.softmax(-1).unsqueeze(1).repeat(8, 300, 1)
        )[0].transpose(0,1)
        
        query_embeds = init_query_feature + query_embeds
        query_embeds = self.catext_norm(query_embeds)

        (
            inter_states,
            init_reference,
            inter_references,
            enc_outputs_class,
            enc_outputs_coord_unact,
        ) = self.transformer(
            multi_level_feats, 
            multi_level_masks, 
            multi_level_position_embeddings, 
            query_embeds,
            dataset_name = dataset_name
            # dataset_name = 'o365'
        )

        # inter_states_label = F.normalize(self.temp(inter_states),p=2, dim=-1)
        inter_states_label = self.temp(inter_states)

        # Calculate output coordinates and classes.
        outputs_classes = []
        outputs_coords = []
        for lvl in range(inter_states.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)

            # from label_embedding.debug.s365_label_embedding_4 import o365 as obb365
            # from label_embedding.debug.s365_label_embedding_4_list import o365 as obb365
            # from label_embedding.o365_label_embedding_addnone_4 import o365 
            # from label_embedding.coco_label_embedding_addnone_4 import coco80 as coco_LE
            # coco_LE = torch.tensor(coco_LE, device='cuda')
            # o365 = torch.tensor(o365, device='cuda')
            # outputs_class =inter_states_label[lvl]@o365.T
            outputs_class = self.class_embed[dataset_name][lvl](inter_states_label[lvl])
            tmp = self.bbox_embed[lvl](inter_states[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        # tensor shape: [num_decoder_layers, bs, num_query, num_classes]
        outputs_coord = torch.stack(outputs_coords)
        # tensor shape: [num_decoder_layers, bs, num_query, 4]

        # prepare for loss computation
        output = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        if self.aux_loss:
            output["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)

        if self.as_two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            output["enc_outputs"] = {
                "pred_logits": enc_outputs_class,
                "pred_boxes": enc_outputs_coord,
            }

        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)
            
            loss_dict = self.criterion(output, targets, dataset_name)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
                    
            # online sampling
            self.losslog[dataset_name].append({"loss_bbox":loss_dict['loss_bbox'].item(),\
                                                "loss_class":loss_dict['loss_class'].item()})
            self.count += 1
            if self.count == 2000 and self.online_sample:
                log_path = Path(os.path.join(self.output_dir,'losslog/loss.json')) if \
                    self.output_dir != None else Path('losslog/loss.json')
                device = torch.cuda.current_device()
                # if device == 0 and os.uname().nodename=='ai_hgx_02':
                if device == 0:
                    import json
                    if not os.path.exists(str(log_path.parent)):
                        os.mkdir(str(log_path.parent))
                    with open(log_path,'w')as f:
                        json.dump(self.losslog, f)
                self.count = 0
                self.losslog = {'coco':[],'lvis_v1':[],'o365':[],'oid':[]}                   

            return loss_dict
        else:
            # gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            # targets = self.prepare_targets(gt_instances)
            # loss_dict = self.criterion(output, targets, dataset_name)
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            results = self.inference(box_cls, box_pred, images.image_sizes)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]

    def inference(self, box_cls, box_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        # Select top-k confidence boxes for inference
        prob = box_cls.sigmoid()
        topk_values, topk_indexes = torch.topk(
            prob.view(box_cls.shape[0], -1), self.select_box_nums_for_evaluation, dim=1
        )
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, box_cls.shape[2], rounding_mode="floor")
        labels = topk_indexes % box_cls.shape[2]

        boxes = torch.gather(box_pred, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(
            zip(scores, labels, boxes, image_sizes)
        ):
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))
            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)
        return results

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            new_targets.append({"labels": gt_classes, "boxes": gt_boxes})
        return new_targets

    def preprocess_image(self, batched_inputs):
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images)
        return images

