from mmengine.config import read_base
from seg.models.utils.dsa import DSA_V14


with read_base():
    from ..resnet.fcn_r18_d8_40k_flare22 import * # noqa

model.update(dict(
    backbone=dict(
        final_plugins=dict(type=DSA_V14))))

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='flare22', name='fcn-r18-dsnet-v14'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(
    type=SegLocalVisualizer, vis_backends=vis_backends, name='visualizer')

# optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0005)
# optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
