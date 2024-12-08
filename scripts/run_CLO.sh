# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tools/train_net.py --config-file projects/dino/configs/dino-dinov2/dino_dinov2_4scale_12ep_LS.py --num-gpus 8
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python tools/train_net.py --config-file projects/dino/configs/dino-dinov2/dino_dinov2_4scale_12ep_LS.py --num-gpus 6 --resume
# export LRU_CACHE_CAPACITY=1
# export NCCL_P2P_DISABLE=1
# export OPENBLAS_NUM_THREADS=1
export PYTHONWARNINGS="ignore"
python tools/train_net.py --config-file projects/deformable_detr/configs/deformable_detr_r50_two_stage_clo.py --num-gpus 8 --resume
# python tools/train_net.py --config-file projects/deformable_detr/configs/holi.py --num-gpus 8 
