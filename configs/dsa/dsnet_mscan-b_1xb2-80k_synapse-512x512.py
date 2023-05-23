_base_ = [
    '../segnext/segnext-b_1xb2-80k_synapse-512x512.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
# model settings
model = dict(
    decode_head=dict(
        _delete_=True,
        type='MSDSNHead',
        in_channels=[128, 320, 512],
        in_index=[1, 2, 3],
        channels=512,
        dropout_ratio=0.1,
        num_classes=9,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.2),
            dict(
                type='DiceLoss', loss_weight=0.8)]))

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='synapse', name='dsn-segnext-160k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(type='SegLocalVisualizer',
                  vis_backends=vis_backends,
                  name='visualizer')

randomness = dict(seed=None)
