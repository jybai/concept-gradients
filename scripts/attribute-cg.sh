#!/bin/bash

set -e

DSET=$1
ARCH=$2
SAVE_DIR="./attrs/${DSET}"

mkdir -p $SAVE_DIR

case $ARCH in 
    
    inception_v3)
        X2Y_PATH="./models/${DSET}/${DSET}_x2y_inception_v3_all-data/version_0/model.ckpt"
        LAYERS="Mixed_6c Mixed_6d Mixed_6e Mixed_7a Mixed_7b Mixed_7c fc"
        
        for LAYER in $LAYERS
        do
            X2C_PATH="./models/${DSET}/${DSET}_x2c_inception_v3_ft-x2y_ft-${LAYER}+/version_0/model.ckpt"
            SAVE_PATH=$SAVE_DIR"/cg_inception_v3_${LAYER}+.npy"
            python -m cg.attribute_cg $DSET $X2Y_PATH $X2C_PATH $SAVE_PATH --x2y_arch_name inception_v3 --layer $LAYER
        done
        
        X2C_PATH="./models/${DSET}/${DSET}_x2c_inception_v3_ft-x2y/version_0/model.ckpt"
        SAVE_PATH=$SAVE_DIR"/cg_inception_v3_input+.npy"
        python -m cg.attribute_cg $DSET $X2Y_PATH $X2C_PATH $SAVE_PATH --x2y_arch_name inception_v3
        ;;
    
    resnet50)
        X2Y_PATH="./models/${DSET}/${DSET}_x2y_resnet50_all-data/version_0/model.ckpt"
        LAYERS="layer3.3 layer3.4 layer3.5 layer4.0 layer4.1 layer4.2 fc"
        
        for LAYER in $LAYERS
        do
            X2C_PATH="./models/${DSET}/${DSET}_x2c_resnet50_ft-x2y_ft-${LAYER}+/version_0/model.ckpt"
            SAVE_PATH=$SAVE_DIR"/cg_resnet50_${LAYER}+.npy"
            python -m cg.attribute_cg $DSET $X2Y_PATH $X2C_PATH $SAVE_PATH --x2y_arch_name resnet50 --layer $LAYER
        done
        
        X2C_PATH="./models/${DSET}/${DSET}_x2c_resnet50_ft-x2y/version_0/model.ckpt"
        SAVE_PATH=$SAVE_DIR"/cg_resnet50_input+.npy"
        python -m cg.attribute_cg $DSET $X2Y_PATH $X2C_PATH $SAVE_PATH --x2y_arch_name resnet50
        ;;
        
    vgg16_bn)
        X2Y_PATH="./models/${DSET}/${DSET}_x2y_vgg16_bn_all-data/version_0/model.ckpt"
        LAYERS="features.7 features.14 features.24 features.34 classifier.0 classifier.3 classifier.6"
        
        for LAYER in $LAYERS
        do
            X2C_PATH="./models/${DSET}/${DSET}_x2c_vgg16_bn_ft-x2y_ft-${LAYER}+/version_0/model.ckpt"
            SAVE_PATH=$SAVE_DIR"/cg_vgg16_bn_${LAYER}+.npy"
            python -m cg.attribute_cg $DSET $X2Y_PATH $X2C_PATH $SAVE_PATH --x2y_arch_name vgg16_bn --bsize 16 --layer $LAYER
        done
        
        X2C_PATH="./models/${DSET}/${DSET}_x2c_vgg16_bn_ft-x2y/version_0/model.ckpt"
        SAVE_PATH=$SAVE_DIR"/cg_vgg16_bn_input+.npy"
        python -m cg.attribute_cg $DSET $X2Y_PATH $X2C_PATH $SAVE_PATH --x2y_arch_name vgg16_bn --bsize 16
        ;;
    
    *)
        echo -n "unknown arch"
        ;;
esac
