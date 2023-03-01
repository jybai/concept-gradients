#!/bin/bash

set -e

DSET=$1
ARCH=$2
SAVE_DIR="./attrs/${DSET}"

mkdir -p $SAVE_DIR

case $ARCH in 
    
    inception_v3)

        X2Y_PATH="./models/${DSET}/${DSET}_x2y_inception_v3_all-data/version_0/model.ckpt"
        CAV_DIR="./cavs/${DSET}/${ARCH}"
        python -m cg.attribute_cav $DSET $X2Y_PATH $CAV_DIR --base_save_path $SAVE_DIR --arch_name $ARCH
        ;;
    
    resnet50)

        X2Y_PATH="./models/${DSET}/${DSET}_x2y_resnet50_all-data/version_0/model.ckpt"
        CAV_DIR="./cavs/${DSET}/${ARCH}"
        python -m cg.attribute_cav $DSET $X2Y_PATH $CAV_DIR --base_save_path $SAVE_DIR --arch_name $ARCH
        ;;
        
    vgg16_bn)

        X2Y_PATH="./models/${DSET}/${DSET}_x2y_vgg16_bn_all-data/version_0/model.ckpt"
        CAV_DIR="./cavs/${DSET}/${ARCH}"
        python -m cg.attribute_cav $DSET $X2Y_PATH $CAV_DIR --base_save_path $SAVE_DIR --arch_name $ARCH
        ;;
        
    *)
        echo -n "unknown arch"
        ;;
esac
