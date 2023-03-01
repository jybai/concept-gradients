#!/bin/bash

set -e

ARCH=$1

case $ARCH in 
    inception_v3)
        python -m cg.train_cg 'cuba-unvoted' 'x2y' --arch_name inception_v3 --lr 0.01 --scheduler_step 20 --weight_decay 0.00004 --use_all_data --save_model
        ;;

    resnet50)
        python -m cg.train_cg 'cuba-unvoted' 'x2y' --arch_name resnet50 --lr 0.01 --scheduler_step 15 --weight_decay 0.0004 --use_all_data --save_model
        ;;

    vgg16_bn)
        python -m cg.train_cg 'cuba-unvoted' 'x2y' --arch_name vgg16_bn --lr 0.001 --scheduler_step 15 --weight_decay 0.0004 --bsize 32 --use_all_data --save_model
        ;;

    *)
        echo -n "unknown arch"
        ;;
esac
