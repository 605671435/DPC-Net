# import modules
from torch.nn import SyncBatchNorm
from mmseg.models import SegDataPreProcessor
from mmseg.models.segmentors import EncoderDecoder
from mmseg.models.backbones import ResNetV1c
from mmseg.models.decode_heads import FCNHead
from mmseg.models.losses import CrossEntropyLoss, DiceLoss
from mmengine.model.weight_init import PretrainedInit
# model settings
norm_cfg = dict(type=SyncBatchNorm, requires_grad=True)
data_preprocessor = dict(
    type=SegDataPreProcessor,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type=EncoderDecoder,
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type=ResNetV1c,
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True,
        init_cfg=dict(
            type=PretrainedInit, checkpoint='open-mmlab://resnet50_v1c')),
    decode_head=dict(
        type=FCNHead,
        in_channels=2048,
        in_index=3,
        channels=512,
        num_convs=2,
        concat_input=True,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type=CrossEntropyLoss, use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type=FCNHead,
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type=CrossEntropyLoss, use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
