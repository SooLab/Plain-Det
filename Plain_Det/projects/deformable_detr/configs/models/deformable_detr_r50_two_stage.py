import torch.nn as nn
import torch

from detectron2.modeling.backbone import ResNet, BasicStem
from detectron2.layers import ShapeSpec
from detectron2.config import LazyCall as L

from detrex.modeling.matcher import HungarianMatcher
from detrex.modeling.neck import ChannelMapper
from detrex.layers import PositionEmbeddingSine
from LVIS_V1_1203_label_embedding import lvis1203 as lvis_v1_LE 
from object365_label_embedding import object365 as ob365
from coco_80_label_embedding import coco80 as coco_LE
from label_embedding.oid_label_embedding_the_object import oid500

from projects.deformable_detr.modeling import (
    DeformableDETR,
    DeformableDetrTransformerEncoder,
    DeformableDetrTransformerDecoder,
    DeformableDetrTransformer,
    DeformableCriterion,
)

lvis_v1_LE = torch.tensor(lvis_v1_LE, device='cuda')
coco_LE = torch.tensor(coco_LE, device='cuda')
o365 = torch.tensor(ob365, device='cuda')
oid = torch.tensor(oid500, device='cuda')
LE = [coco_LE, lvis_v1_LE, o365, oid]

model = L(DeformableDETR)(
    label_embedding=LE,
    backbone=L(ResNet)(
        stem=L(BasicStem)(in_channels=3, out_channels=64, norm="FrozenBN"),
        stages=L(ResNet.make_default_stages)(
            depth=50,
            stride_in_1x1=False,
            norm="FrozenBN",
        ),
        out_features=["res3", "res4", "res5"],
        freeze_at=1,
    ),
    position_embedding=L(PositionEmbeddingSine)(
        num_pos_feats=256,
        temperature=10000,
        normalize=True,
        offset=-0.5,
    ),
    neck=L(ChannelMapper)(
        input_shapes={
            "res3": ShapeSpec(channels=512),
            "res4": ShapeSpec(channels=1024),
            "res5": ShapeSpec(channels=2048),
        },
        in_features=["res3", "res4", "res5"],
        out_channels=512,
        num_outs=4,
        kernel_size=1,
        norm_layer=L(nn.GroupNorm)(num_groups=32, num_channels=512),
    ),
    transformer=L(DeformableDetrTransformer)(
        encoder=L(DeformableDetrTransformerEncoder)(
            embed_dim=512,
            num_heads=8,
            feedforward_dim=1024,
            attn_dropout=0.1,
            ffn_dropout=0.1,
            num_layers=6,
            post_norm=False,
            num_feature_levels="${..num_feature_levels}",
        ),
        decoder=L(DeformableDetrTransformerDecoder)(
            embed_dim=512,
            num_heads=8,
            feedforward_dim=1024,
            attn_dropout=0.1,
            ffn_dropout=0.1,
            num_layers=6,
            return_intermediate=True,
            num_feature_levels="${..num_feature_levels}",
        ),
        as_two_stage="${..as_two_stage}",
        num_feature_levels=4,
        two_stage_num_proposals="${..num_queries}",
        learnt_init_query = False,
    ),
    embed_dim=512,
    num_classes=[80, 1203, 365, 500],
    num_queries=300,
    aux_loss=True,
    with_box_refine=False,
    as_two_stage=False,
    criterion=L(DeformableCriterion)(
        num_classes=1203,
        matcher=L(HungarianMatcher)(
            cost_class=2.0,
            cost_bbox=5.0,
            cost_giou=2.0,
            cost_class_type="focal_loss_cost",
            alpha=0.25,
            gamma=2.0,
        ),
        weight_dict={
            "loss_class": 1.0,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
        },
        loss_class_type="focal_loss",
        alpha=0.25,
        gamma=2.0,
    ),
    pixel_mean=[123.675, 116.280, 103.530],
    pixel_std=[58.395, 57.120, 57.375],
    select_box_nums_for_evaluation=300,
    device="cuda",
)

# set aux loss weight dict
if model.aux_loss:
    weight_dict = model.criterion.weight_dict
    aux_weight_dict = {}
    for i in range(model.transformer.decoder.num_layers - 1):
        aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
    aux_weight_dict.update({k + "_enc": v for k, v in weight_dict.items()})
    weight_dict.update(aux_weight_dict)
    model.criterion.weight_dict = weight_dict