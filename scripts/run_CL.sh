export PYTHONWARNINGS="ignore"
python tools/train_net.py --config-file projects/deformable_detr/configs/deformable_detr_r50_two_stage_90k_cocolvis.py --num-gpus 8 --resume
