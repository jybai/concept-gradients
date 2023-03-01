#!/bin/bash

set -e

DSET=$1
ARCH=$2
SAVE_DIR="./attrs/${DSET}"

mkdir -p $SAVE_DIR

X2Y_PATH="./models/${DSET}/${DSET}_x2y_${ARCH}_all-data/version_0/model.ckpt"
SAVE_PATH="${SAVE_DIR}/ce_x2y_${ARCH}.npy"
python -m cg.attribute_ce $DSET x2y $X2Y_PATH $SAVE_PATH --arch_name $ARCH

X2C_PATH="./models/${DSET}/${DSET}_x2c_${ARCH}_ft-x2y/version_0/model.ckpt"
SAVE_PATH="${SAVE_DIR}/ce_x2c_${ARCH}_input+.npy"
python -m cg.attribute_ce $DSET x2c $X2C_PATH $SAVE_PATH --arch_name $ARCH

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
    X2C_PATH="./models/${DSET}/${DSET}_x2c_${ARCH}_ft-x2y_ft-${LAYER}+/version_0/model.ckpt"
    SAVE_PATH="${SAVE_DIR}/ce_x2c_${ARCH}_${LAYER}+.npy"
    python -m cg.attribute_ce $DSET x2c $X2C_PATH $SAVE_PATH --arch_name $ARCH --layer $LAYER
done
