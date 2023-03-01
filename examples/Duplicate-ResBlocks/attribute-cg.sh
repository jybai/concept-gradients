#!/bin/bash

set -e

DSET=$1
DCOPIES=$2
DLAYER=$3
DBLOCK=$4

ARCH="dup-resnet50"
SAVE_DIR="../../scripts/attrs/${DSET}"
mkdir -p $SAVE_DIR

LAYER="layer${DLAYER}.${DBLOCK}"

X2Y_PATH="../../scripts/models/${DSET}/${DSET}_x2y_resnet50_all-data/version_0/model.ckpt"

X2C_DIR="../../scripts/models/${DSET}/${DSET}_x2c_${ARCH}_ft-x2y_ft-${LAYER}+_${DCOPIES}x"
X2C_PATH="${X2C_DIR}/version_0/model.ckpt"
CFG_PATH="${X2C_DIR}/version_0/hparams.yaml"

SAVE_PATH="${SAVE_DIR}/cg_${ARCH}_${LAYER}+_${DCOPIES}x.npy"

python -m cg.attribute_cg $DSET $X2Y_PATH $X2C_PATH $SAVE_PATH \
--x2y_arch_name resnet50 --x2c_arch_name $ARCH \
--layer $LAYER --x2c_layer "${LAYER}.0" \
--x2c_cfg_path $CFG_PATH
