#!/bin/sh
set -e

export CUDA_VISIBLE_DEVICES="2,3"

for LR in '1e-3' '1e-4' '1e-5'
do
  for WD in '1e-2' '1e-3' '1e-4'
  do 
  SAVE_PATH="./logs/psphead_freeze-pretrained_lr${LR}_wd${WD}"
  python train.py --save_path $SAVE_PATH --load_ckpt_path "/home/andrewbai/data/ChestX_Det/pspnet_chestxray_best_model_4.pkl" --freeze_pretrained --lr $LR --weight_decay $WD
  done
done
