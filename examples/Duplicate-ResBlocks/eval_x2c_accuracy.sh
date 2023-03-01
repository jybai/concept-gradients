#!/bin/bash

set -e

DSET=$1
DCOPIES=$2
DLAYER=$3
DBLOCK=$4

ARCH="dup-resnet50"
LAYER="layer${DLAYER}.${DBLOCK}"

MODEL_PATH="../../scripts/models/${DSET}/${DSET}_x2c_${ARCH}_ft-x2y_ft-${LAYER}+_${DCOPIES}x"
X2C_PATH="${MODEL_PATH}/version_0/model.ckpt"
CFG_PATH="${MODEL_PATH}/version_0/hparams.yaml"

python -m cg.eval_x2c_accuracy $DSET $X2C_PATH --x2c_arch_name $ARCH --model_cfg_path $CFG_PATH
