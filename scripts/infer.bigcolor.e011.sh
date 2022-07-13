#!/bin/bash

set -e

dir_target="$(pwd)/results/bigcolor"

# INFERENCE
python -W ignore colorize.py \
    --path_ckpt './ckpts/bigcolor' \
    --path_output $dir_target \
    --epoch 11 \
    --use_ema 
