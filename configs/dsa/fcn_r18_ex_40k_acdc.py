_base_ = ['../resnet/fcn_r18-d8_1xb2-40k_acdc-256x256.py']

model = dict(
    backbone=dict(
        plugins=[dict(cfg=dict(type='DSA', attn_types=('sp', 'ch'), fusion_type='dsa', bias=True),
                      stages=(False, False, False, True),
                      position='after_conv2'),
                 ]
    )
)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='acdc', name='fcn-r34-ex-40k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')

