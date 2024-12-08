# export NCCL_P2P_DISABLE=1
# export PYTHONWARNINGS="ignore"
# python tools/train_net.py --config-file projects/deformable_detr/configs/deformable_detr_r50_two_stage_800k_clod.py  --num-gpus 8 --eval-only train.init_checkpoint=YOUR_WEIGHT_HERE
python tools/train_net.py --config-file projects/deformable_detr/configs/deformable_detr_r50_two_stage_800k_clod.py  --num-gpus 8 --eval-only train.init_checkpoint=/inspurfs/group/yangsb/zhuyuchen/ECCV2024/detrex/CLOD_BS64_CLSA_800k/model_final.pth