from mmengine.config import read_base
from seg.models.utils import DSA

with read_base():
    from ..resnet.faster_rcnn_r50_fpn_1x_coco import *  # noqa

model.update(dict(
    backbone=dict(
        frozen_stages=1,
        init_cfg=dict(type='Pretrained', prefix='backbone', checkpoint='work_dirs/ckpts/resnet_ex_in1k/top1_77-28_epoch_98.pth'),
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
            project='coco', name='r50-ex'))
]
visualizer = dict(type=DetLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
