#!/bin/bash

set -e

RECALL_TYPE=$1

DSET_TRN="cuba-unvoted"
DSET_VAL="cuba-unvoted"
ARCH="inception_v3"
LAYER="Mixed_7c"
ATTR_DIR="./attrs/${DSET_TRN}"

MODES="chain_rule_joint chain_rule_independent cav inner_prod cosine_similarity"

for MODE in $MODES
do
    ATTR_NPY_PATH=$ATTR_DIR"/cg_${ARCH}_${LAYER}+_mode-${MODE}.npy"
    echo $MODE
    case $RECALL_TYPE in
        sample)
            python -m cg.eval_recall_sample $ATTR_NPY_PATH $DSET_VAL
            ;;
        class)
            python -m cg.eval_recall_class $ATTR_NPY_PATH $DSET_VAL
            ;;
        *)
            echo -n "unknown recall type"
            ;;
    esac
done
