from mmengine.config import read_base
from monai.networks.nets import SwinUNETR
from seg.models.decode_heads.decode_head import LossHead
with read_base():
    from .._base_.models.unet_r18_d16_ds import *  # noqa
    from .._base_.datasets.flare22 import *  # noqa
    from .._base_.schedules.schedule_40k import *  # noqa
    from .._base_.default_runtime import *  # noqa
# model settings
crop_size = (512, 512)
data_preprocessor.update(dict(size=crop_size))
model.update(
    dict(data_preprocessor=data_preprocessor,
         pretrained=None,
         neck=None,
         decode_head=dict(num_classes=14),
         auxiliary_head=None,
         test_cfg=dict(mode='whole')))
model['backbone'] = dict(
    type=SwinUNETR,
    img_size=crop_size,
    feature_size=48,
    in_channels=1,
    out_channels=14,
    spatial_dims=2)
model['decode_head'] = dict(
    type=LossHead,
    num_classes=14,
    loss_decode=[
        dict(
            type=CrossEntropyLoss, use_sigmoid=False, loss_weight=1.0),
        dict(
            type=MemoryEfficientSoftDiceLoss, loss_weight=1.0)])

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='flare22', name='swin-unetr-40k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
