#!/usr/bin/env bash

## Initialize our own variables:
EXPERIMENT=''
RUN_VERSION=''
SRC_VERSION=''
SRC_ITER=''
GPU_COUNT=0

while getopts ":e:v:f:i:g:" opt; do
    case "$opt" in
    e)  EXPERIMENT=$OPTARG
        ;;
    v)  RUN_VERSION=$OPTARG
        ;;
    f)  SRC_VERSION=$OPTARG
        ;;
    i)  SRC_ITER=$OPTARG
        ;;
    g)  GPU_COUNT=$OPTARG
        ;;
    esac
done


echo "EXPERIMENT=${EXPERIMENT}, RUN_VERSION=${RUN_VERSION}, SRC_VERSION=${SRC_VERSION}, SRC_ITER=${SRC_ITER}, GPU_COUNT=${GPU_COUNT}"


# setup
pip install -r requirements.txt
mkdir -p /detectron/lib/datasets/data/logs/
mkdir -p /detectron/lib/datasets/data/results/${RUN_VERSION}/


# Test
nohup python2 tools/test_net.py \
    --multi-gpu-testing \
    --vis \
    --wait False \
    --cfg configs/nuclei/${EXPERIMENT}.yaml \
    --output-folder /detectron/lib/datasets/data/results/${RUN_VERSION}/ \
    VIS True \
    NUM_GPUS ${GPU_COUNT} \
    >> /detectron/lib/datasets/data/logs/test_log 2>&1 &


## Test all snapshots in a folder
#nohup python2 tools/test_net.py \
#    --multi-gpu-testing \
#    --vis \
#    --wait False \
#    --cfg configs/nuclei/${EXPERIMENT}.yaml \
#    --output-folder /detectron/lib/datasets/data/results/${RUN_VERSION}/ \
#    --weights-folder /detectron/lib/datasets/data/models/${RUN_VERSION}/ \
#    VIS True \
#    NUM_GPUS ${GPU_COUNT} \
#    >> /detectron/lib/datasets/data/logs/test_log 2>&1 &
