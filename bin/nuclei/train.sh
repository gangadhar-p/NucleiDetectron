#!/usr/bin/env bash

# setup
pip install -r requirements.txt
mkdir -p /detectron/lib/datasets/data/logs/

## Initialize our own variables:
EXPERIMENT=''
RUN_VERSION=''

while getopts ":e:v:g:" opt; do
    case "$opt" in
    e)  EXPERIMENT=$OPTARG
        ;;
    v)  RUN_VERSION=$OPTARG
        ;;
    g)  GPU_COUNT=$OPTARG
        ;;
    esac
done

echo "EXPERIMENT=${EXPERIMENT}, RUN_VERSION=${RUN_VERSION} GPU_COUNT=${GPU_COUNT}"

# train
nohup python2 tools/train_net.py \
    --multi-gpu-testing \
    --skip-test \
    --cfg configs/nuclei/${EXPERIMENT}.yaml \
    OUTPUT_DIR /detectron/lib/datasets/data/models/${RUN_VERSION} \
    NUM_GPUS ${GPU_COUNT} \
    >> /detectron/lib/datasets/data/logs/train_log 2>&1 &