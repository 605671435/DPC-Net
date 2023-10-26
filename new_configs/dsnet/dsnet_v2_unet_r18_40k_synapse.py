from mmengine.config import read_base
from seg.models.utils.dsa_v2 import DSA


with read_base():
    from ..unet.unet_r18_d16_ds_40k_synapse import * # noqa

model['neck'].update(dict(fusion_cfg=dict(type=DSA, ratio=16)))

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='dsnet-v2-unet-r18'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(
    type=SegLocalVisualizer, vis_backends=vis_backends, name='visualizer')

# optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0005)
# optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
