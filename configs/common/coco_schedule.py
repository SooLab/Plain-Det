from fvcore.common.param_scheduler import MultiStepParamScheduler

from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler


def default_X_scheduler(num_X):
    """
    Returns the config for a default multi-step LR scheduler such as "1x", "3x",
    commonly referred to in papers, where every 1x has the total length of 1440k
    training images (~12 COCO epochs). LR is decayed twice at the end of training
    following the strategy defined in "Rethinking ImageNet Pretraining", Sec 4.
    Args:
        num_X: a positive real number
    Returns:
        DictConfig: configs that define the multiplier for LR during training
    """
    # total number of iterations assuming 16 batch size, using 1440000/16=90000
    total_steps_16bs = num_X * 90000

    if num_X <= 2:
        scheduler = L(MultiStepParamScheduler)(
            values=[1.0, 0.1, 0.01],
            # note that scheduler is scale-invariant. This is equivalent to
            # milestones=[6, 8, 9]
            milestones=[60000, 80000, 90000],
        )
    else:
        scheduler = L(MultiStepParamScheduler)(
            values=[1.0, 0.1, 0.01],
            milestones=[total_steps_16bs - 60000, total_steps_16bs - 20000, total_steps_16bs],
        )
    return L(WarmupParamScheduler)(
        scheduler=scheduler,
        warmup_length=1000 / total_steps_16bs,
        warmup_method="linear",
        warmup_factor=0.001,
    )


def default_coco_scheduler(epochs=50, decay_epochs=40, warmup_epochs=0):
    """
    Returns the config for a default multi-step LR scheduler such as "50epochs",
    commonly referred to in papers, where every 1x has the total length of 1440k
    training images (~12 COCO epochs). LR is decayed once at the end of training.

    Args:
        epochs (int): total training epochs.
        decay_epochs (int): lr decay steps.
        warmup_epochs (int): warmup epochs.

    Returns:
        DictConfig: configs that define the multiplier for LR during training
    """
    # total number of iterations assuming 16 batch size, using 1440000/16=90000
    total_steps_16bs = epochs * 7500
    decay_steps = decay_epochs * 7500
    warmup_steps = warmup_epochs * 7500
    scheduler = L(MultiStepParamScheduler)(
        values=[1.0, 0.1],
        milestones=[decay_steps, total_steps_16bs],
    )
    return L(WarmupParamScheduler)(
        scheduler=scheduler,
        warmup_length=warmup_steps / total_steps_16bs,
        warmup_method="linear",
        warmup_factor=0.001,
    )

def default_lvis_scheduler(epochs=50, decay_epochs=40, warmup_epochs=0):
    """
    Returns the config for a default multi-step LR scheduler such as "50epochs",
    commonly referred to in papers, where every 1x has the total length of 1440k
    training images (~12 COCO epochs). LR is decayed once at the end of training.

    Args:
        epochs (int): total training epochs.
        decay_epochs (int): lr decay steps.
        warmup_epochs (int): warmup epochs.

    Returns:
        DictConfig: configs that define the multiplier for LR during training
    """
    # total number of iterations assuming 16 batch size, using 1440000/16=90000
    total_steps_16bs = epochs * 3200
    decay_steps = decay_epochs * 3200
    warmup_steps = warmup_epochs * 3200
    scheduler = L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[130000, 138000, 140000],
        # milestones=[300000, 450000, 500000],
    )
    return L(WarmupParamScheduler)(
        scheduler=scheduler,
        warmup_length=warmup_steps / total_steps_16bs,
        warmup_method="linear",
        warmup_factor=0.001,
    )

def default_scheduler(epochs=50, decay_epochs=40, warmup_epochs=0):
    """
    Returns the config for a default multi-step LR scheduler such as "50epochs",
    commonly referred to in papers, where every 1x has the total length of 1440k
    training images (~12 COCO epochs). LR is decayed once at the end of training.

    Args:
        epochs (int): total training epochs.
        decay_epochs (int): lr decay steps.
        warmup_epochs (int): warmup epochs.

    Returns:
        DictConfig: configs that define the multiplier for LR during training
    """
    # total number of iterations assuming 16 batch size, using 1440000/16=90000
    total_steps_16bs = epochs * 3200
    decay_steps = decay_epochs * 3200
    warmup_steps = warmup_epochs * 3200
    scheduler = L(MultiStepParamScheduler)(
        values=[1.0, 1.0, 1.0],
        milestones=[300000, 350000, 500000],
    )
    return L(WarmupParamScheduler)(
        scheduler=scheduler,
        warmup_length=warmup_steps / total_steps_16bs,
        warmup_method="linear",
        warmup_factor=0.001,
    )
def default_o365_scheduler(epochs=50, decay_epochs=40, warmup_epochs=0):
    """
    Returns the config for a default multi-step LR scheduler such as "50epochs",
    commonly referred to in papers, where every 1x has the total length of 1440k
    training images (~12 COCO epochs). LR is decayed once at the end of training.

    Args:
        epochs (int): total training epochs.
        decay_epochs (int): lr decay steps.
        warmup_epochs (int): warmup epochs.

    Returns:
        DictConfig: configs that define the multiplier for LR during training
    """
    # total number of iterations assuming 16 batch size, using 1440000/16=90000
    total_steps_16bs = epochs * 3200
    decay_steps = decay_epochs * 3200
    warmup_steps = warmup_epochs * 3200
    scheduler = L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[340000, 390000, 400000],
    )
    return L(WarmupParamScheduler)(
        scheduler=scheduler,
        warmup_length=warmup_steps / total_steps_16bs,
        warmup_method="linear",
        warmup_factor=0.001,
    )
    
    
def custom_scheduler(decay, total, mid=0, warmup_epochs=0):
    """
    Returns the config for a default multi-step LR scheduler such as "50epochs",
    commonly referred to in papers, where every 1x has the total length of 1440k
    training images (~12 COCO epochs). LR is decayed once at the end of training.

    Args:
        epochs (int): total training epochs.
        decay_epochs (int): lr decay steps.
        warmup_epochs (int): warmup epochs.

    Returns:
        DictConfig: configs that define the multiplier for LR during training
    """
    # total number of iterations assuming 16 batch size, using 1440000/16=90000
    total_steps_16bs = total
    decay_steps = decay
    warmup_steps = warmup_epochs * 3200
    if mid == 0:
        scheduler = L(MultiStepParamScheduler)(
            values=[1.0, 0.1],
            milestones=[decay, total],
        )
    else:
        scheduler = L(MultiStepParamScheduler)(
            values=[1.0, 0.1, 0.01],
            milestones=[decay, mid, total],
        )
    return L(WarmupParamScheduler)(
        scheduler=scheduler,
        warmup_length=warmup_steps / total_steps_16bs,
        warmup_method="linear",
        warmup_factor=0.001,
    )
# default coco scheduler
lr_multiplier_1x = default_X_scheduler(1)
lr_multiplier_2x = default_X_scheduler(2)
lr_multiplier_3x = default_X_scheduler(3)
lr_multiplier_6x = default_X_scheduler(6)
lr_multiplier_9x = default_X_scheduler(9)


# default scheduler for detr
lr_multiplier_50ep = default_coco_scheduler(50, 40, 0)
lr_multiplier_36ep = default_coco_scheduler(36, 30, 0)
lr_multiplier_24ep = default_coco_scheduler(24, 20, 0)
lr_multiplier_12ep = default_coco_scheduler(12, 11, 0)
lr_exp = default_scheduler()
# warmup scheduler for detr
lr_multiplier_50ep_warmup = default_coco_scheduler(50, 40, 1e-3)
lr_multiplier_12ep_warmup = default_coco_scheduler(12, 11, 1e-3)
lvis_v1_coco_lr_multiplier_50ep_warmup = default_lvis_scheduler(50, 45, 1e-3)
lr_o365_warmup = default_o365_scheduler(50, 45, 1e-3)

lr_lvis_test = custom_scheduler(85000,90000)
lr_coco_test = custom_scheduler(40000,45000)
lr_CL_test = custom_scheduler(173000,180000)
lr_oid_test = custom_scheduler(300000,350000)
lr_eruption = custom_scheduler(999,1000)
lr_finetune = custom_scheduler(4,5)
lr_CLOD_800k = custom_scheduler(740000,800000,790000)
lr_CLO_400k = custom_scheduler(360000,400000,395000)
lr_COv1D2019 = custom_scheduler(660000,720000,700000)
lr_CLOD_swin = custom_scheduler(1600000,1700000,1680000)
lr_tempp=custom_scheduler(200000,250000,240000)