#!/bin/bash

python -W ignore ./colorize_multi_c.py --classes 15 11 14 88 100 \
                                         --path_ckpt ckpts/bigcolor \
                                         --epoch 11 \
                                         --seed -1 \
                                         --path_input ./images_multi_c/ \
                                         --path_output results_multi_c
