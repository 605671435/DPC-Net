dataset_type = 'ACDC'
data_root = 'data/acdc/'
img_scale = (256, 256)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Normalize'),
    dict(type='Resize', scale=img_scale, keep_ratio=False),
    dict(type='RandomRotFlip', rotate_prob=0.5, flip_prob=0.5, degree=20),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Normalize'),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
# train/val list from https://github.com/282857341/UNet-2022/blob/master/nnunet/network_configuration/config.py
ACDC_train_list = ['patient001_frame01', 'patient001_frame12', 'patient004_frame01', 'patient004_frame15',
                   'patient005_frame01', 'patient005_frame13', 'patient006_frame01', 'patient006_frame16',
                   'patient007_frame01', 'patient007_frame07', 'patient010_frame01', 'patient010_frame13',
                   'patient011_frame01', 'patient011_frame08', 'patient013_frame01', 'patient013_frame14',
                   'patient015_frame01', 'patient015_frame10', 'patient016_frame01', 'patient016_frame12',
                   'patient018_frame01', 'patient018_frame10', 'patient019_frame01', 'patient019_frame11',
                   'patient020_frame01', 'patient020_frame11', 'patient021_frame01', 'patient021_frame13',
                   'patient022_frame01', 'patient022_frame11', 'patient023_frame01', 'patient023_frame09',
                   'patient025_frame01', 'patient025_frame09', 'patient026_frame01', 'patient026_frame12',
                   'patient027_frame01', 'patient027_frame11', 'patient028_frame01', 'patient028_frame09',
                   'patient029_frame01', 'patient029_frame12', 'patient030_frame01', 'patient030_frame12',
                   'patient031_frame01', 'patient031_frame10', 'patient032_frame01', 'patient032_frame12',
                   'patient033_frame01', 'patient033_frame14', 'patient034_frame01', 'patient034_frame16',
                   'patient035_frame01', 'patient035_frame11', 'patient036_frame01', 'patient036_frame12',
                   'patient037_frame01', 'patient037_frame12', 'patient038_frame01', 'patient038_frame11',
                   'patient039_frame01', 'patient039_frame10', 'patient040_frame01', 'patient040_frame13',
                   'patient041_frame01', 'patient041_frame11', 'patient043_frame01', 'patient043_frame07',
                   'patient044_frame01', 'patient044_frame11', 'patient045_frame01', 'patient045_frame13',
                   'patient046_frame01', 'patient046_frame10', 'patient047_frame01', 'patient047_frame09',
                   'patient050_frame01', 'patient050_frame12', 'patient051_frame01', 'patient051_frame11',
                   'patient052_frame01', 'patient052_frame09', 'patient054_frame01', 'patient054_frame12',
                   'patient056_frame01', 'patient056_frame12', 'patient057_frame01', 'patient057_frame09',
                   'patient058_frame01', 'patient058_frame14', 'patient059_frame01', 'patient059_frame09',
                   'patient060_frame01', 'patient060_frame14', 'patient061_frame01', 'patient061_frame10',
                   'patient062_frame01', 'patient062_frame09', 'patient063_frame01', 'patient063_frame16',
                   'patient065_frame01', 'patient065_frame14', 'patient066_frame01', 'patient066_frame11',
                   'patient068_frame01', 'patient068_frame12', 'patient069_frame01', 'patient069_frame12',
                   'patient070_frame01', 'patient070_frame10', 'patient071_frame01', 'patient071_frame09',
                   'patient072_frame01', 'patient072_frame11', 'patient073_frame01', 'patient073_frame10',
                   'patient074_frame01', 'patient074_frame12', 'patient075_frame01', 'patient075_frame06',
                   'patient076_frame01', 'patient076_frame12', 'patient077_frame01', 'patient077_frame09',
                   'patient078_frame01', 'patient078_frame09', 'patient080_frame01', 'patient080_frame10',
                   'patient082_frame01', 'patient082_frame07', 'patient083_frame01', 'patient083_frame08',
                   'patient084_frame01', 'patient084_frame10', 'patient085_frame01', 'patient085_frame09',
                   'patient086_frame01', 'patient086_frame08', 'patient087_frame01', 'patient087_frame10']
ACDC_val_list = ['patient089_frame01', 'patient089_frame10', 'patient090_frame04', 'patient090_frame11',
                 'patient091_frame01', 'patient091_frame09', 'patient093_frame01', 'patient093_frame14',
                 'patient094_frame01', 'patient094_frame07', 'patient096_frame01', 'patient096_frame08',
                 'patient097_frame01', 'patient097_frame11', 'patient098_frame01', 'patient098_frame09',
                 'patient099_frame01', 'patient099_frame09', 'patient100_frame01', 'patient100_frame13']

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        case_list=ACDC_train_list,
        data_prefix=dict(
            img_path='img_dir/train', seg_map_path='ann_dir/train'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        case_list=ACDC_val_list,
        data_prefix=dict(img_path='img_dir/train', seg_map_path='ann_dir/train'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mDice', 'mIoU'])
test_evaluator = dict(type='CaseMetric', iou_metrics=['mDice'], hd_metric=True, ignore_index=0)
