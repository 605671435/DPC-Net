from mmengine.config import read_base
from seg.models.utils.dsa import DSA_V3


with read_base():
    from ..unet.unet_r18v1c_d8_40k_flare22 import * # noqa


model.update(dict(
    backbone=dict(
        final_plugins=dict(type=DSA_V3, fusion_type='dam'))))


vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='flare22', name='unet-r18v1c-v3-dam-40k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(
    type=SegLocalVisualizer, vis_backends=vis_backends, name='visualizer')

