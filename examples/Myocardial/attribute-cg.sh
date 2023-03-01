#!/bin/bash

set -e

X2Y_PATH="./myocardial_target.pth"
X2C_PATH="./myocardial_concept.pth"

python attribute_cg.py $X2Y_PATH $X2C_PATH --layer "head.0"
