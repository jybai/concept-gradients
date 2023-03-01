#!/bin/bash

set -e

LAYER=$1
RECALL_TYPE=$2

DSET="cuba-unvoted"
ARCH="dup-resnet50"

for DCOPIES in 2 3 4
do
    ATTR_NPY_PATH="../../scripts/attrs/${DSET}/cg_${ARCH}_${LAYER}+_${DCOPIES}x.npy"

    case $RECALL_TYPE in
        sample)
            python -m cg.eval_recall_sample $ATTR_NPY_PATH $DSET
            ;;
        class)
            python -m cg.eval_recall_class $ATTR_NPY_PATH $DSET
            ;;
        *)
            echo -n "unknown recall type"
            ;;
    esac
done
