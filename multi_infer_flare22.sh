#!/usr/bin/env bash

CONFIGS=(
#'new_configs/unet/unet_r18v1c_d8_40k_flare22.py'
#'new_configs/se/unet_r18v1c_se_40k_flare22.py'
#'new_configs/cbam/unet_r18v1c_cbam_40k_flare22.py'
#'new_configs/psa/unet_r18v1c_psa_s_40k_flare22.py'
#'new_configs/ccnet/unet_r18v1c_ccnet_40k_flare22.py'
#'new_configs/gcnet/unet_r18v1c_gcb_40k_flare22.py'
#'new_configs/hamnet/unet_r18v1c_hamnet_40k_flare22.py'
#'new_configs/dsnet/unet_r18v1c_dsnet_v14_40k_flare22.py'
#'new_configs/attn_unet/attn_ma_unet_r18v1c_flare22_40k.py'
'new_configs/medical_seg/swin_unetr_base_40k_flare22.py'
'new_configs/medical_seg/missformer_40k_flare22.py'
'new_configs/medical_seg/mednext_40k_flare22.py'
'new_configs/dsnet/unet_r18v1c_dsnet_v14_dam_40k_flare22.py'
)
WORK_DIRS=(
#'vis_ckpts/unet_r18v1c_d8_40k_flare22/best_mDice_86-44_iter_40000.pth'
#'work_dirs/unet_r18v1c_se_40k_flare22/5-run_20231018_123046/run2/best_mDice_86-72_iter_40000.pth'
#'vis_ckpts/unet_r18v1c_cbam_40k_flare22/best_mDice_86-40_iter_40000.pth'
#'vis_ckpts/unet_r18v1c_psa_s_40k_flare22/best_mDice_86-76_iter_40000.pth'
#'vis_ckpts/unet_r18v1c_ccnet_40k_flare22/best_mDice_86-68_iter_40000.pth'
#'vis_ckpts/unet_r18v1c_gcb_40k_flare22/best_mDice_87-16_iter_40000.pth'
#'vis_ckpts/unet_r18v1c_hamnet_40k_flare22/best_mDice_87-06_iter_40000.pth'
#'vis_ckpts/unet_r18v1c_dsnet_v14_40k_flare22/best_mDice_87-33_iter_40000.pth'
#'vis_ckpts/attn_ma_unet_r18v1c_flare22_40k/best_mDice_91-00_iter_28000.pth'
'vis_ckpts/swin_unetr_base_40k_flare22/best_mDice_85-86_iter_40000.pth'
'vis_ckpts/missformer_40k_flare22/best_mDice_85-15_iter_40000.pth'
'vis_ckpts/mednext_40k_flare22/best_mDice_86-60_iter_40000.pth'
'vis_ckpts/unet_r18v1c_dsnet_v14_dam_40k_flare22/best_mDice_87-14_iter_40000.pth'
)
IMGS=(
'/home/jz207/workspace/data/FLARE22/img_dir/val/case0004/case0004_slice046.jpg'
'/home/jz207/workspace/data/FLARE22/img_dir/val/case0008/case0008_slice053.jpg'
'/home/jz207/workspace/data/FLARE22/img_dir/val/case0010/case0010_slice050.jpg'
'/home/jz207/workspace/data/FLARE22/img_dir/val/case0014/case0014_slice020.jpg'
'/home/jz207/workspace/data/FLARE22/img_dir/val/case0015/case0015_slice036.jpg'
'/home/jz207/workspace/data/FLARE22/img_dir/val/case0016/case0016_slice041.jpg'
'/home/jz207/workspace/data/FLARE22/img_dir/val/case0018/case0018_slice058.jpg'
'/home/jz207/workspace/data/FLARE22/img_dir/val/case0020/case0020_slice048.jpg'
'/home/jz207/workspace/data/FLARE22/img_dir/val/case0024/case0024_slice055.jpg'
'/home/jz207/workspace/data/FLARE22/img_dir/val/case0025/case0025_slice036.jpg'
'/home/jz207/workspace/data/FLARE22/img_dir/val/case0027/case0027_slice053.jpg'
)
# 使用 for in 循环遍历数组元素
for IMG in "${IMGS[@]}"
do
#  PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
#    python tools/visualizations/vis_gt.py $IMG
  i=0
  for CONFIG in "${CONFIGS[@]}"
  do
    echo $CONFIG
    echo ${WORK_DIRS[$i]}
    echo $IMG
    PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
    python tools/inference.py \
      $CONFIG \
      ${WORK_DIRS[$i]} \
      $IMG
    i=$i+1
  done
done
