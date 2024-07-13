from detrex.config import get_config
from .models.deformable_detr_r50_256dim import model

dataloader = get_config("common/data/CLOD_detr.py").dataloader
lr_multiplier = get_config("common/coco_schedule.py").lr_CLOD_800k
optimizer = get_config("common/optim.py").AdamW
train = get_config("common/train.py").train

# modify training config
train.init_checkpoint = "weight/resnet50_miil_21k_modified2.pkl"
train.output_dir = "outputs/CLOD_BS64_CLSA_800k"


# max training iterations
train.max_iter = 800000 #50 epoch 64B  

# run evaluation every 5000 iters
train.eval_period = 5000000

# log training infomation every 20 iters
train.log_period = 50

# save checkpoint every 5000 iters
train.checkpointer.period = 50000

# gradient clipping for training
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2be

# set training devices
train.device = "cuda"
model.device = train.device
model.with_box_refine = True
model.as_two_stage = True
model.output_dir = train.output_dir
model.online_sample = True

# modify optimizer config
optimizer.lr = 1e-4
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

# modify dataloader config
dataloader.train.num_workers = 2
dataloader.online_sample = model.online_sample
# please notice that this is total batch size.
# each gpu is 64/16 = 4
dataloader.train.total_batch_size = 64

# dump the testing results into output_dir for visualization
dataloader.evaluator.output_dir = train.output_dir
