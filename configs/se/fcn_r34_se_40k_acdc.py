_base_ = ['../resnet/fcn_r34-d8_1xb2-40k_acdc-256x256.py']

model = dict(
    backbone=dict(plugins=[dict(cfg=dict(type='SELayer'),
                                stages=(True, True, True, True),
                                position='after_conv1')]))

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='acdc', name='fcn-r34-se-40k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')

