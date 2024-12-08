export PYTHONWARNINGS="ignore"
python tools/train_net.py --config-file projects/deformable_detr/configs/deformable_detr_r50_two_stage_clo.py --num-gpus 8 --resume
