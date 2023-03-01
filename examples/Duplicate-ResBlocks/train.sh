#!/bin/bash

set -e

DCOPIES=$1
DLAYER=$2
DBLOCK=$3

DSET="cuba-unvoted"
HPARAMS="--lr 0.01 --scheduler_step 25 --weight_decay 0.00004 --save_model"
X2Y_CKPT="../../scripts/models/${DSET}/${DSET}_x2y_resnet50_all-data/version_0/model.ckpt"
SAVE_DIR="../../scripts/models/${DSET}/"

python trainer.py $DSET 'x2c' $HPARAMS --x2c_from_x2y_ckpt $X2Y_CKPT --save_dir $SAVE_DIR --duplicate_copies $DCOPIES --duplicate_layer $DLAYER --duplicate_block $DBLOCK
