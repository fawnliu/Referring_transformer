now=$(date +"%Y%m%d_%H%M%S")

python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --use_env \
    main.py \
    --model HTA_small \
    --batch-size 64  \
    --dist-eval \
    --drop-path 0.3 \
    --output_dir HTA_small \
    --data-path /home/work2/lf/Datasets/ILSVRC/Data/CLS-LOC/


# wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash