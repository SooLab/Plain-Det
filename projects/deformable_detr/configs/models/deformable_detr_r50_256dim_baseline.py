import torch.nn as nn
import torch

from detectron2.modeling.backbone import ResNet, BasicStem
from detectron2.layers import ShapeSpec
from detectron2.config import LazyCall as L

from detrex.modeling.matcher import HungarianMatcher
from detrex.modeling.neck import ChannelMapper
from detrex.layers import PositionEmbeddingSine
from label_embedding.lvis_label_embedding_addnone_4 import lvis1203 as lvis_v1_LE 
from label_embedding.coco_label_embedding_addnone_4 import coco80 as coco_LE
from label_embedding.o365_label_embedding_addnone_4 import o365 
from label_embedding.oid_label_embedding_addnone_4 import oid500
from label_embedding.OID_v4_601_label_embedding_4 import oid601

from projects.deformable_detr.modeling import (
    DeformableDETR,
    DeformableDETR256B,
    DeformableDetrTransformerEncoderB,
    DeformableDetrTransformerDecoderB,
    DeformableDetrTransformerB,
    DeformableCriterion,
)

lvis_v1_LE = torch.tensor(lvis_v1_LE, device='cuda')
coco_LE = torch.tensor(coco_LE, device='cuda')
oid = torch.tensor(oid601, device='cuda')
o365 = torch.tensor(o365, device='cuda')
LE = [coco_LE, lvis_v1_LE, o365, oid]

model = L(DeformableDETR256B)(
    output_dir = None,
    online_sample = False,
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
        # num_pos_feats=256,
        num_pos_feats=128,
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
        out_channels=256,
        num_outs=4,
        kernel_size=1,
        norm_layer=L(nn.GroupNorm)(num_groups=32, num_channels=256),
    ),
    transformer=L(DeformableDetrTransformerB)(
        encoder=L(DeformableDetrTransformerEncoderB)(
            embed_dim=256,
            num_heads=8,
            feedforward_dim=1024,
            attn_dropout=0.1,
            ffn_dropout=0.1,
            num_layers=6,
            post_norm=False,
            num_feature_levels="${..num_feature_levels}",
        ),
        decoder=L(DeformableDetrTransformerDecoderB)(
            embed_dim=256,
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
        learnt_init_query = True,
    ),
    embed_dim=256,
    num_classes=[80, 1203, 365, 601],
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
