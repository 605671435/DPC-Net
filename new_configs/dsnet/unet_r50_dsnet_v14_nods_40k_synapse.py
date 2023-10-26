from mmengine.config import read_base
from seg.models.utils.dsa import DSA_V14


with read_base():
    from ..unet.unet_r50_d16_40k_synapse import * # noqa

model.update(dict(
    backbone=dict(
        norm_cfg=dict(type=InstanceNorm2d, requires_grad=True),
        final_plugins=dict(type=DSA_V14)),
    neck=dict(
        norm_cfg=dict(type=InstanceNorm2d, requires_grad=True),
        act_cfg=dict(type=LeakyReLU)),
    decode_head=dict(
        norm_cfg=dict(type=InstanceNorm2d, requires_grad=True))))

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='unet-r50-V14-nods-40k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(
    type=SegLocalVisualizer, vis_backends=vis_backends, name='visualizer')

# optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0005)
# optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
