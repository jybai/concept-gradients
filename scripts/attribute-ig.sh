#!/bin/bash

set -e
export CUDA_VISIBLE_DEVICES

DSET=$1
SAVE_DIR="./attrs/${DSET}"

mkdir -p $SAVE_DIR

for ARCH in 'vgg16_bn' # 'inception_v3' 'resnet50'
do
  python -m cg.attribute_ig $DSET "./models/${DSET}/${DSET}_x2y_${ARCH}_all-data/version_0/model.ckpt" "${SAVE_DIR}/ig_${ARCH}.npy" --bsize 4 --x2y_arch_name $ARCH 
  # bsize=8 takes 46GB vram for inception_v3, bsize=4 takes 39 GB vram for vgg16_bn
done

