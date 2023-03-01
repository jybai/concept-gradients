#!/bin/bash

set -e

CUB_SRC_DIR="/home/andrewbai/data/CUB_200_2011" # $1
CUB_CBM_DIR="/home/andrewbai/data/CUB_processed/class_attr_data_10" # $3

CUB_RAW_DIR="/home/andrewbai/data/CUB_raw" # $2
CUB_UNVOTED_DIR="/home/andrewbai/data/CUB_unvoted" # $4

mkdir -p $CUB_RAW_DIR
mkdir -p $CUB_UNVOTED_DIR

python -m cg.CUB.data_processing -save_dir $CUB_RAW_DIR -data_dir $CUB_SRC_DIR --ref_data_dir $CUB_CBM_DIR
python -m cg.CUB.generate_new_data "CreateClassAttributesData" --out_dir $CUB_UNVOTED_DIR --data_dir $CUB_RAW_DIR --min_class_count 10 --keep_instance_data

