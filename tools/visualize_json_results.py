#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import json
import numpy as np
import os
from collections import defaultdict
import cv2
import tqdm

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import Boxes, BoxMode, Instances
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from Plain_Det.data.datasets.object365 import register_object365_instances
from detectron2.data.datasets import register_coco_instances, register_lvis_instances
from Plain_Det.data.datasets.register_oid import register_oid_instances
from Plain_Det.data.datasets.register_oid_classic import register_oid_instances_classic
from Plain_Det.data.datasets import categories
from Plain_Det.data.datasets import oidv4
from Plain_Det.evaluation import OIDEvaluator
DATASET_PATH = '/public/home/zhuyuchen530/projects/detrex/datasets'
RFS_PATH = os.path.join(DATASET_PATH,"rfs","oid_rfs.json")

def _get_builtin_metadata(cats):
    id_to_name = {x['id']: x['name'] for x in cats}
    thing_dataset_id_to_contiguous_id = {i + 1: i for i in range(len(cats))}
    thing_classes = [x['name'] for x in sorted(cats, key=lambda x: x['id'])]
    return {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes}
    
_CLASSIC_REGISTER_OID = {
    "oid_val": ("/storage/data/zhuyuchen530/oid/images/validation/", "/storage/data/zhuyuchen530/oid/annotations/openimages_challenge_2019_val_v2.json"),
    "oid_val_expanded": ("/storage/data/zhuyuchen530/oid/images/validation/", "/storage/data/zhuyuchen530/oid/annotations/openimages_challenge_2019_val_v2_expanded.json"),
    "oid_train": ("/storage/data/zhuyuchen530/oid/images", "/storage/data/zhuyuchen530/oid/annotations/oid_challenge_2019_train_bbox.json"),
    "oidv4_val_expanded": ("/storage/data/zhuyuchen530/oid/images/validation/", "/inspurfs/group/yangsb/zhuyuchen/datasets/OID_V4/annotations/openimages_v4_val_bbox.json"),
}
for key, (image_root, json_file) in _CLASSIC_REGISTER_OID.items():
    register_oid_instances_classic(
        key,
        _get_builtin_metadata(oidv4),
        json_file,
        image_root,
    )
register_object365_instances("object365_train", {}, "/public/home/zhuyuchen530/projects/detrex/datasets/object365v2/annotations/zhiyuan_objv2_train.json", "/public/home/zhuyuchen530/projects/detrex/datasets/object365v2/images/train")
register_object365_instances("object365_val", {}, "/public/home/zhuyuchen530/projects/detrex/datasets/object365v2/annotations/zhiyuan_objv2_val.json", "/public/home/zhuyuchen530/projects/detrex/datasets/object365v2/images/val")
register_coco_instances("coco_2017_val_0", {}, "/public/home/zhuyuchen530/projects/detrex/datasets/coco/annotations/instances_val2017.json", "/public/home/zhuyuchen530/projects/detrex/datasets/coco/val2017")
def create_instances(predictions, image_size):
    ret = Instances(image_size)

    score = np.asarray([x["score"] for x in predictions])
    chosen = (score > args.conf_threshold).nonzero()[0]
    score = score[chosen]
    bbox = np.asarray([predictions[i]["bbox"] for i in chosen]).reshape(-1, 4)
    bbox = BoxMode.convert(bbox, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)

    labels = np.asarray([dataset_id_map(predictions[i]["category_id"]) for i in chosen])

    ret.scores = score
    ret.pred_boxes = Boxes(bbox)
    ret.pred_classes = labels

    try:
        ret.pred_masks = [predictions[i]["segmentation"] for i in chosen]
    except KeyError:
        pass
    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script that visualizes the json predictions from COCO or LVIS dataset."
    )
    parser.add_argument("--input", required=True, help="JSON file produced by the model")
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument("--dataset", help="name of the dataset", default="coco_2017_val")
    parser.add_argument("--conf-threshold", default=0.5, type=float, help="confidence threshold")
    args = parser.parse_args()

    logger = setup_logger()

    with PathManager.open(args.input, "r") as f:
        predictions = json.load(f)

    pred_by_image = defaultdict(list)
    for p in predictions:
        pred_by_image[p["image_id"]].append(p)

    dicts = list(DatasetCatalog.get(args.dataset))
    metadata = MetadataCatalog.get(args.dataset)
    if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):

        def dataset_id_map(ds_id):
            return metadata.thing_dataset_id_to_contiguous_id[ds_id]

    elif "lvis" in args.dataset:
        # LVIS results are in the same format as COCO results, but have a different
        # mapping from dataset category id to contiguous category id in [0, #categories - 1]
        def dataset_id_map(ds_id):
            return ds_id - 1

    else:
        raise ValueError("Unsupported dataset: {}".format(args.dataset))

    os.makedirs(args.output, exist_ok=True)

    for dic in tqdm.tqdm(dicts):
        img = cv2.imread(dic["file_name"], cv2.IMREAD_COLOR)[:, :, ::-1]
        basename = os.path.basename(dic["file_name"])

        predictions = create_instances(pred_by_image[dic["image_id"]], img.shape[:2])
        vis = Visualizer(img, metadata)
        vis_pred = vis.draw_instance_predictions(predictions).get_image()

        vis = Visualizer(img, metadata)
        vis_gt = vis.draw_dataset_dict(dic).get_image()

        concat = np.concatenate((vis_pred, vis_gt), axis=1)
        cv2.imwrite(os.path.join(args.output, basename), concat[:, :, ::-1])
