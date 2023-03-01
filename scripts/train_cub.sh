#!/bin/bash

ADDITIONAL_ARGS=$1
python -m cg.train 'cuba' 'x2y' --save_model $ADDITIONAL_ARGS
