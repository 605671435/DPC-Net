#!/usr/bin/env bash

CONFIGS=(
#'new_configs/unet/unet_r18v1c_d8_40k_synapse.py'
#'new_configs/se/unet_r18v1c_se_40k_synapse.py'
#'new_configs/cbam/unet_r18v1c_cbam_40k_synapse.py'
#'new_configs/psa/unet_r18v1c_psa_s_40k_synapse.py'
#'new_configs/ccnet/unet_r18v1c_ccnet_40k_synapse.py'
#'new_configs/gcnet/unet_r18v1c_gcb_40k_synapse.py'
#'new_configs/attn_unet/attn_ma_unet_r18v1c_synapse_40k.py'
#'new_configs/hamnet/unet_r18v1c_hamnet_40k_synapse.py'
#'new_configs/dsnet/unet_r18v1c_dsnet_v14_40k_synapse.py'
#'new_configs/medical_seg/swin_unetr_base_40k_synapse.py'
#'new_configs/medical_seg/missformer_40k_synapse.py'
#'new_configs/medical_seg/mednext_40k_synapse.py'
'new_configs/dsnet/unet_r18v1c_dsnet_v14_dam_40k_synapse.py'
)
WORK_DIRS=(
#'vis_ckpts/unet_r18v1c_d8_40k_synapse/best_mDice_73-87_iter_40000.pth'
#'vis_ckpts/unet_r18v1c_se_40k_synapse/best_mDice_75-51_iter_40000.pth'
#'vis_ckpts/unet_r18v1c_cbam_40k_synapse/best_mDice_75-72_iter_36000.pth'
#'vis_ckpts/unet_r18v1c_psa_s_40k_synapse/best_mDice_75-76_iter_40000.pth'
#'vis_ckpts/unet_r18v1c_ccnet_40k_synapse/best_mDice_76-59_iter_40000.pth'
#'vis_ckpts/unet_r18v1c_gcb_40k_synapse/best_mDice_76-16_iter_40000.pth'
#'work_dirs/attn_ma_unet_r18v1c_synapse_40k/5-run_20231013_165701/run0/best_mDice_77-77_iter_36000.pth'
#'vis_ckpts/unet_r18v1c_hamnet_40k_synapse/best_mDice_77-43_iter_40000.pth'
#'vis_ckpts/unet_r18v1c_dsnet_v14_40k_synapse/best_mDice_79-30_iter_36000.pth'
#'vis_ckpts/swin_unetr_base_40k_synapse/best_mDice_77-11_iter_36000.pth'
#'vis_ckpts/missformer_40k_synapse/best_mDice_78-24_iter_40000.pth'
#'vis_ckpts/mednext_40k_synapse/best_mDice_77-60_iter_36000.pth'
'vis_ckpts/unet_r18v1c_dsnet_v14_dam_40k_synapse/best_mDice_77-88_iter_40000.pth'
)
#IMGS=(
#'/home/jz207/workspace/zhangdw/ex_kd/data/synapse9/img_dir/val/case0001/case0001_slice094.jpg'
#'/home/jz207/workspace/zhangdw/ex_kd/data/synapse9/img_dir/val/case0002/case0002_slice087.jpg'
#'/home/jz207/workspace/zhangdw/ex_kd/data/synapse9/img_dir/val/case0004/case0004_slice106.jpg'
#'/home/jz207/workspace/zhangdw/ex_kd/data/synapse9/img_dir/val/case0022/case0022_slice063.jpg'
#'/home/jz207/workspace/zhangdw/ex_kd/data/synapse9/img_dir/val/case0036/case0036_slice130.jpg'
#)
IMGS=(
'/home/jz207/workspace/zhangdw/ex_kd/data/synapse9/img_dir/val/case0001/case0001_slice094.jpg'
'/home/jz207/workspace/zhangdw/ex_kd/data/synapse9/img_dir/val/case0002/case0002_slice087.jpg'
'/home/jz207/workspace/zhangdw/ex_kd/data/synapse9/img_dir/val/case0004/case0004_slice106.jpg'
'/home/jz207/workspace/zhangdw/ex_kd/data/synapse9/img_dir/val/case0022/case0022_slice063.jpg'
'/home/jz207/workspace/zhangdw/ex_kd/data/synapse9/img_dir/val/case0036/case0036_slice130.jpg'
'/home/jz207/workspace/zhangdw/ex_kd/data/synapse9/img_dir/val/case0001/case0001_slice112.jpg'
'/home/jz207/workspace/zhangdw/ex_kd/data/synapse9/img_dir/val/case0002/case0002_slice110.jpg'
'/home/jz207/workspace/zhangdw/ex_kd/data/synapse9/img_dir/val/case0022/case0022_slice066.jpg'
'/home/jz207/workspace/zhangdw/ex_kd/data/synapse9/img_dir/val/case0032/case0032_slice108.jpg'
'/home/jz207/workspace/zhangdw/ex_kd/data/synapse9/img_dir/val/case0036/case0036_slice138.jpg'
'/home/jz207/workspace/zhangdw/ex_kd/data/synapse9/img_dir/val/case0032/case0032_slice112.jpg'
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
