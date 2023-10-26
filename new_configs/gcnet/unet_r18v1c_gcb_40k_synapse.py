from mmengine.config import read_base
from mmcv.cnn.bricks import ContextBlock


with read_base():
    from ..unet.unet_r18v1c_d8_40k_synapse import * # noqa

model.update(dict(
    backbone=dict(
        plugins=[dict(
            cfg=dict(type=ContextBlock, ratio=1. / 4),
            stages=(False, True, True, True),
            position='after_conv2')]
    )
))

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='unet-r18v1c-gcb'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(
    type=SegLocalVisualizer, vis_backends=vis_backends, name='visualizer')
