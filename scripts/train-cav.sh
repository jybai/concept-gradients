#!/bin/bash

set -e

DSET=$1
ARCH=$2

case $ARCH in 
    
    inception_v3)
        X2Y_PATH="./models/${DSET}/${DSET}_x2y_inception_v3_all-data/version_0/model.ckpt"
        LAYERS="Mixed_6c Mixed_6d Mixed_6e Mixed_7a Mixed_7b Mixed_7c fc"
        SAVE_DIR="./cavs/${DSET}/${ARCH}"
        mkdir -p $SAVE_DIR
        python -m cg.train_tcav $DSET $X2Y_PATH --arch_name $ARCH --layers $LAYERS --save_dir $SAVE_DIR --force
        ;;
        
    resnet50)
        X2Y_PATH="./models/${DSET}/${DSET}_x2y_resnet50_all-data/version_0/model.ckpt"
        LAYERS="layer3.3 layer3.4 layer3.5 layer4.0 layer4.1 layer4.2 fc"
        SAVE_DIR="./cavs/${DSET}/${ARCH}"
        mkdir -p $SAVE_DIR
        python -m cg.train_tcav $DSET $X2Y_PATH --arch_name $ARCH --layers $LAYERS --save_dir $SAVE_DIR --force
        ;;
        
    vgg16_bn)
        X2Y_PATH="./models/${DSET}/${DSET}_x2y_vgg16_bn_all-data/version_0/model.ckpt"
        LAYERS="features.7 features.14 features.24 features.34 classifier.0 classifier.3 classifier.6"
        SAVE_DIR="./cavs/${DSET}/${ARCH}"
        mkdir -p $SAVE_DIR
        python -m cg.train_tcav $DSET $X2Y_PATH --arch_name $ARCH --layers $LAYERS --save_dir $SAVE_DIR --force
        ;;
        
    *)
        echo -n "unknown arch"
        ;;
esac