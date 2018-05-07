#!/usr/bin/env bash


python lib/datasets/nuclei/write_submission.py \
    --results-root /detectron/lib/datasets/data/results/ \
    --run-version '1_aug_gray_1_5_1_stage_2_v1' \
    --iters '65999' \
    --area-thresh 15 \
    --acc-thresh 0.9 \
    --intersection-thresh 0.3
