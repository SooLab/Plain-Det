from omegaconf import OmegaConf
import os
import json

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.evaluation import COCOEvaluator
from detectron2.evaluation import LVISEvaluator

from Plain_Det.data import DetrDatasetMapper
from Plain_Det.data import build_custom_train_loader
from Plain_Det.data import MultiDatasetSampler
from Plain_Det.data import get_detection_dataset_dicts_with_source
from Plain_Det.data.datasets.register_oid import register_oid_instances
from Plain_Det.data.datasets.register_oid_classic import register_oid_instances_classic
from Plain_Det.data.datasets.object365 import register_object365_instances
from Plain_Det.data.datasets.object365_classic import register_object365_instances_classic
from Plain_Det.data.datasets import register_lvis_instances, register_lvis_instances_classic
from detectron2.data.datasets import register_coco_instances
from Plain_Det.data.datasets import oidv4
from Plain_Det.data.datasets import oidv6
from Plain_Det.evaluation import OIDEvaluator
DATASET_PATH = '/public/home/zhuyuchen530/projects/detrex/datasets'
RFS_PATH_d = os.path.join(DATASET_PATH,"rfs","oid_rfs.json")
CLSA_PATH_d = os.path.join(DATASET_PATH,"rfs","oidv4_clsaware.json")
RFS_PATH_o = os.path.join(DATASET_PATH,"rfs","object365_rfs.json")
RFS_PATH_l = os.path.join(DATASET_PATH,"rfs","lvis_rfs.json")
register_coco_instances("coco_2017_train_0", {}, "/public/home/zhuyuchen530/projects/detrex/datasets/coco/annotations/instances_train2017.json", "/public/home/zhuyuchen530/projects/detrex/datasets/coco/train2017")
register_coco_instances("coco_2017_val_0", {}, "/public/home/zhuyuchen530/projects/detrex/datasets/coco/annotations/instances_val2017.json", "/public/home/zhuyuchen530/projects/detrex/datasets/coco/val2017")
register_lvis_instances("lvis_v1_train_0", {}, "/public/home/zhuyuchen530/projects/detrex/datasets/lvis/lvis_v1_train_noAnn.json", "/public/home/zhuyuchen530/projects/detrex/datasets/coco/")
register_lvis_instances_classic("lvis_v1_val_0", {}, "/public/home/zhuyuchen530/projects/detrex/datasets/lvis/lvis_v1_val.json", "/public/home/zhuyuchen530/projects/detrex/datasets/coco/")
register_object365_instances("object365_train", {}, "/public/home/zhuyuchen530/projects/detrex/datasets/object365v2/annotations/modified_zhiyuan_objv2_train_noAnn.json", "/public/home/zhuyuchen530/projects/detrex/datasets/object365v2/images/train")
register_object365_instances_classic("object365_val", {}, "/public/home/zhuyuchen530/projects/detrex/datasets/object365v2/annotations/zhiyuan_objv2_val.json", "/public/home/zhuyuchen530/projects/detrex/datasets/object365v2/images/val")

def _get_builtin_metadata(cats):
    id_to_name = {x['id']: x['name'] for x in cats}
    thing_dataset_id_to_contiguous_id = {i + 1: i for i in range(len(cats))}
    thing_classes = [x['name'] for x in sorted(cats, key=lambda x: x['id'])]
    return {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes}

_CLASSIC_REGISTER_OID = {
    # "oid_val": ("/storage/data/zhuyuchen530/oid/images/validation/", "/storage/data/zhuyuchen530/oid/annotations/openimages_challenge_2019_val_v2.json"),
    # "oid_val_expanded": ("/storage/data/zhuyuchen530/oid/images/validation/", "/storage/data/zhuyuchen530/oid/annotations/openimages_challenge_2019_val_v2_expanded.json"),
    "oidv4_val_expanded": ("/storage/data/zhuyuchen530/oid/images/validation/", "/inspurfs/group/yangsb/zhuyuchen/datasets/OID_V4/annotations/openimages_v4_val_bbox.json"),
    # "oidv6_val": ("/storage/data/zhuyuchen530/oid/images/validation/", "/inspurfs/group/yangsb/zhuyuchen/datasets/OID_V6/annotations/openimages_v6_val_bbox.json"),
    # "oid_train": ("/storage/data/zhuyuchen530/oid/images", "/storage/data/zhuyuchen530/oid/annotations/oid_challenge_2019_train_bbox.json"),
}

_TEXT_REGISTER_OID = {
    # "oid_train_txt": ("/storage/data/zhuyuchen530/oid/images", "/storage/data/zhuyuchen530/oid/annotations/oid_challenge_2019_train_bboxnoAnn.json"),
    "oidv4_train_txt": ("/storage/data/zhuyuchen530/oid/images", "/inspurfs/group/yangsb/zhuyuchen/datasets/OID_V4/annotations/openimages_v4_train_bboxnoAnn.json"),
}


for key, (image_root, json_file) in _CLASSIC_REGISTER_OID.items():
    register_oid_instances_classic(
        key,
        _get_builtin_metadata(oidv4),
        # _get_builtin_metadata(oidv6),
        json_file,
        image_root,
    )
for key, (image_root, json_file) in _TEXT_REGISTER_OID.items():
    register_oid_instances(
        key,
        _get_builtin_metadata(oidv4),
        json_file,
        image_root,
    )

dataloader = OmegaConf.create()

dataloader.online_sample = True
dataloader.train = L(build_custom_train_loader)(
    dataset=L(get_detection_dataset_dicts_with_source)(
        # dataset_names=["coco_2017_train_0"],
        dataset_names=["coco_2017_train_0","lvis_v1_train_0", "object365_train", "oidv4_train_txt"],
        # dataset_names=["object365_val","object365_val","object365_train"],
        # dataset_names=["coco_2017_val_0","coco_2017_val_0","coco_2017_val_0","coco_2017_val_0"],
        filter_empty=False,
        ),
    mapper=L(DetrDatasetMapper)(
        augmentation=[
            L(T.RandomFlip)(),
            L(T.ResizeShortestEdge)(
                short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                max_size=1333,
                sample_style="choice",
            ),
        ],
        augmentation_with_crop=[
            L(T.RandomFlip)(),
            L(T.ResizeShortestEdge)(
                short_edge_length=(400, 500, 600),
                sample_style="choice",
            ),
            L(T.RandomCrop)(
                crop_type="absolute_range",
                crop_size=(384, 600),
            ),
            L(T.ResizeShortestEdge)(
                short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                max_size=1333,
                sample_style="choice",
            ),
        ],
        is_train=True,
        mask_on=False,
        img_format="RGB",
    ),
    sampler=L(MultiDatasetSampler)(
        num_workers="${dataloader.train.num_workers}",
        total_batch_size="${dataloader.train.total_batch_size}",
        dataset_dicts="${dataloader.train.dataset}",
        output_dir = "${dataloader.evaluator.output_dir}",
        online_sample = "${dataloader.online_sample}",
        dataset_ratio=[1,1,1,1],
        rfs = {
            # "use_rfs":[True, True, True, True],
            "use_rfs":[True, False, False, False],
            "load_rfs":[False, True, True,True],
            # "load_rfs":[False, False, False,False],
            "load_path":[None,RFS_PATH_l,RFS_PATH_o,CLSA_PATH_d]
        },
        # use_rfs=[True, True, True],
        dataset_ann=["box", "box", "box","box"],
        repeat_threshold=0.001,
    ),
    aspect_ratio_grouping=False, 
    total_batch_size=16,
    num_workers=4,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts_with_source)(dataset_names="coco_2017_val_0", filter_empty=False),
    # dataset=L(get_detection_dataset_dicts_with_source)(dataset_names="lvis_v1_val_0", filter_empty=False),
    # dataset=L(get_detection_dataset_dicts_with_source)(dataset_names="oidv4_val_expanded", filter_empty=False),
    # dataset=L(get_detection_dataset_dicts_with_source)(dataset_names="oidv6_val", filter_empty=False),
    # dataset=L(get_detection_dataset_dicts_with_source)(dataset_names="object365_val", filter_empty=False),
    mapper=L(DetrDatasetMapper)(
        augmentation=[
            L(T.ResizeShortestEdge)(
                short_edge_length=800,
                max_size=1333,
            ),
        ],
        augmentation_with_crop=None,
        is_train=False,
        mask_on=False,
        img_format="RGB",
    ),
    num_workers=1,
)

# dataloader.evaluator = L(OIDEvaluator)(
# dataloader.evaluator = L(LVISEvaluator)(
dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.dataset_names}",
)