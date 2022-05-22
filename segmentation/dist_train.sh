#!/usr/bin/env bash


# CONFIG=configs/upernet_hta_b_512x512_160k_ade20k_swin_setting.py
CONFIG=configs/upernet_hta_s_512x512_160k_ade20k_swin_setting.py
GPUS=8
PORT=${PORT:-29504}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
