# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import sys
import tempfile
import time
import warnings
import cv2
import tqdm

sys.path.insert(0, "./")  # noqa
from demo.predictors import VisualizationDemo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
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
import json
o365 = json.load(open('datasets/metadata/object365_train_cat_info.json','r'))
coco = json.load(open('datasets/coco/annotations/instances_train2017_cat_info.json','r'))
def _get_builtin_metadata(cats):
    id_to_name = {x['id']: x['name'] for x in cats}
    thing_dataset_id_to_contiguous_id = {i + 1: i for i in range(len(cats))}
    thing_classes = [x['name'] for x in sorted(cats, key=lambda x: x['id'])]
    print(thing_classes)
    return {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes}

register_coco_instances("coco_2017_val_0", _get_builtin_metadata(coco), "/public/home/zhuyuchen530/projects/detrex/datasets/coco/annotations/instances_val2017.json", "/public/home/zhuyuchen530/projects/detrex/datasets/coco/val2017")
# register_object365_instances("object365_train", {}, "/public/home/zhuyuchen530/projects/detrex/datasets/object365v2/annotations/modified_zhiyuan_objv2_train_noAnn.json", "/public/home/zhuyuchen530/projects/detrex/datasets/object365v2/images/train")
register_object365_instances_classic("object365_val", _get_builtin_metadata(o365), "/public/home/zhuyuchen530/projects/detrex/datasets/object365v2/annotations/zhiyuan_objv2_val.json", "/public/home/zhuyuchen530/projects/detrex/datasets/object365v2/images/val")
# _CLASSIC_REGISTER_OID = {
#     # "oid_val": ("/storage/data/zhuyuchen530/oid/images/validation/", "/storage/data/zhuyuchen530/oid/annotations/openimages_challenge_2019_val_v2.json"),
#     # "oid_val_expanded": ("/storage/data/zhuyuchen530/oid/images/validation/", "/storage/data/zhuyuchen530/oid/annotations/openimages_challenge_2019_val_v2_expanded.json"),
#     "oidv4_val_expanded": ("/storage/data/zhuyuchen530/oid/images/validation/", "/inspurfs/group/yangsb/zhuyuchen/datasets/OID_V4/annotations/openimages_v4_val_bbox.json"),
#     # "oidv6_val": ("/storage/data/zhuyuchen530/oid/images/validation/", "/inspurfs/group/yangsb/zhuyuchen/datasets/OID_V6/annotations/openimages_v6_val_bbox.json"),
#     # "oid_train": ("/storage/data/zhuyuchen530/oid/images", "/storage/data/zhuyuchen530/oid/annotations/oid_challenge_2019_train_bbox.json"),
# }

# _TEXT_REGISTER_OID = {
#     # "oid_train_txt": ("/storage/data/zhuyuchen530/oid/images", "/storage/data/zhuyuchen530/oid/annotations/oid_challenge_2019_train_bboxnoAnn.json"),
#     "oidv4_train_txt": ("/storage/data/zhuyuchen530/oid/images", "/inspurfs/group/yangsb/zhuyuchen/datasets/OID_V4/annotations/openimages_v4_train_bboxnoAnn.json"),
# }


# for key, (image_root, json_file) in _CLASSIC_REGISTER_OID.items():
#     register_oid_instances_classic(
#         key,
#         _get_builtin_metadata(oidv4),
#         # _get_builtin_metadata(oidv6),
#         json_file,
#         image_root,
#     )
# for key, (image_root, json_file) in _TEXT_REGISTER_OID.items():
#     register_oid_instances(
#         key,
#         _get_builtin_metadata(oidv4),
#         json_file,
#         image_root,
#     )

# constants
WINDOW_NAME = "COCO detections"


def setup(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="detrex demo for visualizing customized inputs")
    parser.add_argument(
        "--config-file",
        default="projects/dino/configs/dino_r50_4scale_12ep.py",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--min_size_test",
        type=int,
        default=800,
        help="Size of the smallest side of the image during testing. Set to zero to disable resize in testing.",
    )
    parser.add_argument(
        "--max_size_test",
        type=float,
        default=1333,
        help="Maximum size of the side of the image during testing.",
    )
    parser.add_argument(
        "--img_format",
        type=str,
        default="RGB",
        help="The format of the loading images.",
    )
    parser.add_argument(
        "--metadata_dataset",
        type=str,
        # default="coco_val_2017_0",
        # default="oidv4_val_expanded",
        # default="lvis_v1_val",
        default="object365_val",
        help="The metadata infomation to be used. Default to COCO val metadata.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup(args)

    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.train.init_checkpoint)

    model.eval()

    demo = VisualizationDemo(
        model=model,
        min_size_test=args.min_size_test,
        max_size_test=args.max_size_test,
        img_format=args.img_format,
        metadata_dataset=args.metadata_dataset,
    )

    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img, args.confidence_threshold)
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                visualized_output.save(out_filename)
            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit
    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        assert args.output is None, "output not yet supported with --webcam!"
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cam.release()
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)
        codec, file_ext = (
            ("x264", ".mkv") if test_opencv_video_format("x264", ".mkv") else ("mp4v", ".mp4")
        )
        if codec == ".mp4v":
            warnings.warn("x264 codec not available, switching to mp4v")
        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + file_ext
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*codec),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()
