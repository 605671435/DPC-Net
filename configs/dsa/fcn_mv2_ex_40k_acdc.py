_base_ = ['../mobilenet_v2/mobilenet_v2_fcn_40k_acdc.py']

model = dict(
    backbone=dict(
        type='MobileNetV2_EX',
        stage_plugin=dict(type='DSA', attn_types=('sp', 'ch'), fusion_type='dsa', bias=True)
))

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='acdc', name='fcn-mv2-ex-40k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')

