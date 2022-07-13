#!/bin/bash

res=256
# INFERENCE
python -W ignore colorize_real.py \
    --path_ckpt 'ckpts/bigcolor' \
    --path_input 'images_gray'\
    --epoch 11 \
    --size_target $res \
    --type_resize 'powerof' \
    --topk 5 \
    --seed -1 \
    --use_ema 
