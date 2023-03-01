#!/bin/sh
set -e
export CUDA_VISIBLE_DEVICES="4,5,6,7"

for LR in '1e-3' '1e-4' '1e-5'
do
  for WD in '1e-2' '1e-3' '1e-4'
  do 
  SAVE_PATH="./logs/psphead_from-scratch_lr${LR}_wd${WD}"
  python train.py --save_path $SAVE_PATH --lr $LR --weight_decay $WD
  done
done
