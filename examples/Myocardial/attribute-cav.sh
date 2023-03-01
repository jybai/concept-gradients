#!/bin/bash

set -e

X2Y_PATH="./myocardial_target.pth"
CAV_DIR="./cavs"

python attribute_cav.py $X2Y_PATH $CAV_DIR
