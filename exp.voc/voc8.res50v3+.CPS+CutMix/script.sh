#!/usr/bin/env bash
nvidia-smi

export volna="path to your data dir"
export NGPUS=4
export OUTPUT_PATH="path to your output dir"
export snapshot_dir=$OUTPUT_PATH/snapshot

export batch_size=8
export learning_rate=0.0025
export snapshot_iter=1

python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py
export TARGET_DEVICE=$[$NGPUS-1]
python eval.py -e 20-34 -d 0-$TARGET_DEVICE --save_path $OUTPUT_PATH/results

# following is the command for debug
# export NGPUS=1
# export batch_size=2
# python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py --debug 1