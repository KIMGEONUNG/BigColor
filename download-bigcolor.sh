#!/bin/bash

# BigColor pretraining model
mkdir ckpts/bigcolor -pv

gdown "1SCrcIxhm2F1SMTUUcNTEtB4fQu6y7RDK" -O ckpts/bigcolor/EG_011.ckpt
gdown "1A6JwWeedJa56_6lDOkkR3hqvJ_aNUghE" -O ckpts/bigcolor/EG_EMA_011.ckpt
gdown "1iQ2OAnQRAqd-6PDtQCrqh5-L_F_GXpq4" -O ckpts/bigcolor/args.pkl
