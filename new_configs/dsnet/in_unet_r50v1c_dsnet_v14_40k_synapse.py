from mmengine.config import read_base
from seg.models.utils.dsa import DSA_V14


with read_base():
    from ..unet.unet_r50v1c_d8_40k_synapse import * # noqa

model.update(dict(
    backbone=dict(
        final_plugins=dict(type=DSA_V14),
        norm_cfg=dict(type=InstanceNorm2d, requires_grad=True)),
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
            project='synapse', name='unet-r18v1c-v14-40k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(
    type=SegLocalVisualizer, vis_backends=vis_backends, name='visualizer')

