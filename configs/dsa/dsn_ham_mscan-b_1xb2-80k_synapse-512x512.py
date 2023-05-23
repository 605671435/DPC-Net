_base_ = [
    '../segnext/segnext-b_1xb2-80k_synapse-512x512.py'
]
# model settings
model = dict(
    backbone=dict(
        type='DSN_MSCAN',
        dsa_stages=(True, True, True, True)))

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='synapse', name='dsn-ham_segnext-160k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(type='SegLocalVisualizer',
                  vis_backends=vis_backends,
                  name='visualizer')
