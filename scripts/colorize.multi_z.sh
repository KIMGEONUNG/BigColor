#!/bin/bash

targets="03830.jpg,76,3.0
16191.jpg,323,5.0
16358.jpg,327,3.0
20557.jpg,411,3.0
20564.jpg,411,3.0
21273.jpg,425,5.0
26932.jpg,538,5.0
32940.jpg,658,3.0
35906.jpg,718,5.0
45630.jpg,912,5.0"

for a in $(echo $targets | xargs -n3); do
    path=$(echo $a | cut -d, -f 1) 
    class=$(echo $a | cut -d, -f 2) 
    std=$(echo $a | cut -d, -f 3) 

    python -W ignore colorize_multi_z.py \
        --path_ckpt './ckpts/bigcolor' \
        --path_output results_multi_z \
        --path_input images_multi_z/$path \
        --idx_class $class \
        --epoch 11 \
        --z_std $std \
        --num_z 20 \
        --use_shuffle \
        --seed -1 \
        --use_ema 
done
    
