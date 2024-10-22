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
from typing import List
import torch

from detrex.modeling import SetCriterion
from detrex.utils import get_world_size, is_dist_avail_and_initialized


class DeformableCriterion(SetCriterion):
    """This class computes the loss for Deformable-DETR
    and two-stage Deformable-DETR
    """

    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        losses: List[str] = ["class", "boxes"],
        eos_coef: float = 0.1,
        loss_class_type: str = "focal_loss",
        alpha: float = 0.25,
        gamma: float = 2.0,
    ):
        super(DeformableCriterion, self).__init__(
            num_classes=num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            losses=losses,
            eos_coef=eos_coef,
            loss_class_type=loss_class_type,
            alpha=alpha,
            gamma=gamma,
        )
        self.count = 0
        self.save_indice_coco = []
        self.save_indice_lvis = []
        self.save_indice_coco_target = []
        self.save_indice_lvis_target = []

    def forward(self, outputs, targets, dataset="lvis_v1"):
        outputs_without_aux = {
            k: v for k, v in outputs.items() if k != "aux_outputs" and k != "enc_outputs"
        }

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # # # for vis****************************
        # current_gpu = torch.cuda.current_device()
        # # 获取GPU设备的名称
        # gpu_name = torch.cuda.get_device_name(current_gpu)

        # # print(f"GPU {current_gpu}: {gpu_name}")

        # # print("this time:",dataset)
        # save_item = []
        # save_target = []

        # for indice in indices:
        #     save_item.append(indice[0].tolist())
        #     print('indice:',indice[1].shape)
        #     save_target.append(indice[1].tolist())
        # if dataset == "lvis_v1":
        #     self.save_indice_lvis.append(save_item)
        #     self.save_indice_lvis_target.append(save_target)
        # if dataset == "coco":
        #     self.save_indice_coco.append(save_item)
        #     self.save_indice_coco_target.append(save_target)
        # print('count:',self.count)
        # if self.count == 5000:
        #     import json
        #     import datetime
        #     import os
        #     current_time = datetime.datetime.now()
        #     time_str = current_time.strftime("%Y-%m-%d_%H-%M")
        #     Floder = f"/public/home/yangsb/project/detrex/vis_Data/{dataset}/{time_str}"
        #     if not os.path.exists(Floder):
        #         os.makedirs(Floder)

        #     if dataset == "coco":
        #         with open(f"/public/home/yangsb/project/detrex/vis_Data/{dataset}/{time_str}/coco_data.py",'w') as file:
        #             json.dump(self.save_indice_coco, file)
        #         with open(f"/public/home/yangsb/project/detrex/vis_Data/{dataset}/{time_str}/coco_data_target.py",'w') as file:
        #             json.dump(self.save_indice_coco_target, file)
        #     if dataset == "lvis_v1":
        #         with open(os.path.join(Floder,'lvis_data.py'),'w') as file2:
        #             json.dump(self.save_indice_lvis, file2)
        #         with open(os.path.join(Floder,'lvis_data_target.py'),'w') as file:
        #             json.dump(self.save_indice_lvis_target, file)
        # self.count += 1
        # # # ***************************************
            

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
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # Compute losses for two-stage deformable-detr
        if "enc_outputs" in outputs:
            enc_outputs = outputs["enc_outputs"]
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt["labels"] = torch.zeros_like(bt["labels"])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {k + "_enc": v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses
