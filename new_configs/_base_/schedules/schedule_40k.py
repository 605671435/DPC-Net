from torch.optim import SGD
from mmengine.optim import OptimWrapper
from mmengine.optim.scheduler import PolyLR
from mmengine.runner.loops import IterBasedTrainLoop, TestLoop, ValLoop
# from seg.engine.runner.loops import ValLoop
from mmengine.hooks import IterTimerHook, LoggerHook, ParamSchedulerHook, DistSamplerSeedHook
from mmseg.engine.hooks import SegVisualizationHook
from seg.engine.hooks import MyCheckpointHook
# optimizer
optimizer = dict(type=SGD, lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type=OptimWrapper, optimizer=optimizer, clip_grad=None)
# learning policy
param_scheduler = [
    dict(
        type=PolyLR,
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=40000,
        by_epoch=False)
]
# training schedule for 40k
train_cfg = dict(type=IterBasedTrainLoop, max_iters=40000, val_interval=4000)
val_cfg = dict(type=ValLoop)
test_cfg = dict(type=TestLoop)
default_hooks = dict(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(type=MyCheckpointHook,
                    by_epoch=False,
                    interval=4000,
                    max_keep_ckpts=1,
                    save_best=['mDice'], rule='greater'),
    sampler_seed=dict(type=DistSamplerSeedHook),
    visualization=dict(type=SegVisualizationHook))
