# #!/usr/bin/env bash

CONFIG=configs/upernet_hta_s_512x512_160k_ade20k_swin_setting.py
CHECKPOINT=work_dirs/upernet_hta_s_512x512_160k_ade20k_swin_setting/checkpoint.pth
GPUS=8

PORT=${PORT:-29500}
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4}  --eval mIoU


# python test.py ${CONFIG} ${CHECKPOINT} --eval mIoU
