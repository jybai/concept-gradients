#!/bin/bash

set -e

ARCH=$1

HPARAMS="--lr 0.01 --scheduler_step 25 --weight_decay 0.00004 --save_model"

case $ARCH in 
    inception_v3)
        X2Y_CKPT="./models/cuba-unvoted_x2y_inception_v3_all-data/version_0/model.ckpt"
        LAYERS="Mixed_6c Mixed_6d Mixed_6e Mixed_7a Mixed_7b Mixed_7c fc"
        python -m cg.train_cg 'cuba-unvoted' 'x2c' $HPARAMS --arch_name inception_v3 --x2c_from_x2y_ckpt $X2Y_CKPT
        for LAYER in $LAYERS
        do
            python -m cg.train_cg 'cuba-unvoted' 'x2c' $HPARAMS --arch_name inception_v3 --x2c_from_x2y_ckpt $X2Y_CKPT --finetune_layer_start $LAYER
        done
        ;;

    resnet50)
        X2Y_CKPT="./models/cuba-unvoted_x2y_resnet50_all-data/version_0/model.ckpt"
        LAYERS="layer3.3 layer3.4 layer3.5 layer4.0 layer4.1 layer4.2 fc"
        python -m cg.train_cg 'cuba-unvoted' 'x2c' $HPARAMS --arch_name resnet50 --x2c_from_x2y_ckpt $X2Y_CKPT
        for LAYER in $LAYERS
        do
            python -m cg.train_cg 'cuba-unvoted' 'x2c' $HPARAMS --arch_name resnet50 --x2c_from_x2y_ckpt $X2Y_CKPT --finetune_layer_start $LAYER
        done
        ;;

    vgg16_bn)
        X2Y_CKPT="./models/cuba-unvoted_x2y_vgg16_bn_all-data/version_0/model.ckpt"
        LAYERS="features.7 features.14 features.24 features.34 classifier.0 classifier.3 classifier.6"
        python -m cg.train_cg 'cuba-unvoted' 'x2c' $HPARAMS --bsize 32 --arch_name vgg16_bn --x2c_from_x2y_ckpt $X2Y_CKPT
        for LAYER in $LAYERS
        do
            python -m cg.train_cg 'cuba-unvoted' 'x2c' $HPARAMS --bsize 32 --arch_name vgg16_bn --x2c_from_x2y_ckpt $X2Y_CKPT --finetune_layer_start $LAYER
        done
        ;;

    *)
        echo -n "unknown arch"
        ;;
esac
