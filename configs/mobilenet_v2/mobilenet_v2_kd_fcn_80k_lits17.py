from mmengine.config import read_base
from seg.models.backbones import MobileNetV2_KD

with read_base():
    from .mobilenet_v2_fcn_40k_acdc import *  # noqa

model.update(dict(
    backbone=dict(type=MobileNetV2_KD)))
