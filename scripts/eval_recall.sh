#!/bin/bash

set -e

DSET_TRN=$1
DSET_VAL=$2
ARCH=$3
METHOD=$4
RECALL_TYPE=$5

case $ARCH in 
    
    inception_v3)
        LAYERS="Mixed_6c Mixed_6d Mixed_6e Mixed_7a Mixed_7b Mixed_7c fc"
        ;;
    
    resnet50)
        LAYERS="layer3.3 layer3.4 layer3.5 layer4.0 layer4.1 layer4.2 fc"
        ;;
        
    vgg16_bn)
        LAYERS="features.7 features.14 features.24 features.34 classifier.0 classifier.3 classifier.6"
        ;;
    
    *)
        echo -n "unknown arch"
        ;;
esac

for LAYER in $LAYERS
do
    case $METHOD in
        cav)
            ATTR_NPY_PATH="./attrs/${DSET_TRN}/cav_${ARCH}_${LAYER}.npy"
            CLASS_KWARGS=""
            ;;
        cg)
            ATTR_NPY_PATH="./attrs/${DSET_TRN}/cg_${ARCH}_${LAYER}+.npy"
            CLASS_KWARGS=""
            ;;
        scg)
            ATTR_NPY_PATH="./attrs/${DSET_TRN}/scg_${ARCH}_${LAYER}+.npy"
            CLASS_KWARGS=""
            ;;
        ce)
            ATTR_NPY_PATH="./attrs/${DSET_TRN}/ce_x2c_${ARCH}_${LAYER}+.npy"
            CLASS_KWARGS="--x2y_npy_path ./attrs/${DSET_TRN}/ce_x2y_${ARCH}.npy --reduce ce_necessary"
            ;;
        *)
            echo -n "unknown method"
            ;;
    esac
    case $RECALL_TYPE in
        sample)
            python -m cg.eval_recall_sample $ATTR_NPY_PATH $DSET_VAL
            ;;
        class)
            python -m cg.eval_recall_class $ATTR_NPY_PATH $DSET_VAL $CLASS_KWARGS
            ;;
        *)
            echo -n "unknown recall type"
            ;;
    esac
done        
