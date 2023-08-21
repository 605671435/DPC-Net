_base_ = ['../resnet/fcn_r50-d8_1xb2-40k_acdc-256x256.py']

model = dict(
    backbone=dict(
        _delete_=True,
        type='MobileNetV2',
        widen_factor=1.,
        strides=(1, 2, 2, 1, 1, 1, 1),
        dilations=(1, 1, 1, 2, 2, 4, 4),
        out_indices=(1, 2, 4, 6),
        norm_cfg=dict(type='SyncBN', requires_grad=True)),
    pretrained=None,
    decode_head=dict(in_channels=320)
)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='acdc', name='fcn-mv2-40k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
