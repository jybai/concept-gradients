#!/bin/bash

set -e

DSET="cuba-unvoted"
ARCH="inception_v3"
LAYER="Mixed_7c"
X2Y_PATH="./models/${DSET}/${DSET}_x2y_inception_v3_all-data/version_0/model.ckpt"

SAVE_DIR="./attrs/${DSET}"
mkdir -p $SAVE_DIR

MODES="chain_rule_joint chain_rule_independent cav inner_prod cosine_similarity"

for MODE in $MODES
do
    X2C_PATH="./models/${DSET}/${DSET}_x2c_inception_v3_ft-x2y_ft-${LAYER}+/version_0/model.ckpt"
    SAVE_PATH=$SAVE_DIR"/cg_inception_v3_${LAYER}+_mode-${MODE}.npy"
    python -m cg.attribute_cg $DSET $X2Y_PATH $X2C_PATH $SAVE_PATH --x2y_arch_name inception_v3 --layer $LAYER --mode $MODE
done
