#!/bin/bash

gpus="0,1,2,3"
size_batch=60
num_epoch=12

CUDA_VISIBLE_DEVICES=$gpus python -W ignore train.py \
                                        --use_enhance \
                                        --coef_enhance 1.2 \
                                        --size_batch $size_batch \
                                        --num_epoch $num_epoch \
                                        --task_name bigcolor_v1 \
                                        --detail "Train for bigcolor" 
