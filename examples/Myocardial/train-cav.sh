#!/bin/bash

set -e

X2Y_PATH="./myocardial_target.pth"
LAYERS="head.0 head.3"
SAVE_DIR="./cavs/"
mkdir -p $SAVE_DIR

python train_cav.py $X2Y_PATH --layers $LAYERS --save_dir $SAVE_DIR --force
