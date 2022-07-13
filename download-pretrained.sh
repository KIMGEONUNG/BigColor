#!/bin/bash

# pretrained model for training

gdown "1usoLZNnDzB3WLAphrGz5W2lJKxMMZJQw" -O pretrained/D_256.pth
gdown "1qvxR2ZK5-gaLgtxem9fdLmfSKKvXF6eW" -O pretrained/config.pickle
gdown "197ypWuWcwKI3Mrq11jbCWkk-r_-cj2tT" -O pretrained/G_ema_256.pth
gdown "1evUHFmNJcFmqlS8YT_SjNfZV3dPjBgPL" -O pretrained/vgg16.pickle
