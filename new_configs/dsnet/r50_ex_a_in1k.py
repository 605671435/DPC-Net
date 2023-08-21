from mmengine.config import read_base
from seg.models.utils import DSA
from seg.models.backbones import ResNet as plugin_resnet
with read_base():
    from ..resnet.resnet50_in1k import * # noqa

model.update(dict(
    backbone=dict(
        type=plugin_resnet,
        plugins=[dict(cfg=dict(type=DSA, attn_types=('sp', 'ch'), fusion_type='dsa', bias=True),
                      stages=(False, False, False, True),
                      position='after_conv2'),
                 ]
    )
))

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='imagenet1k', name='r50-ex-a'))
]
visualizer = dict(type=UniversalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
