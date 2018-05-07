#!/usr/bin/env python2

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Perform inference on one or more datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import os
import pprint
import sys
import time

from caffe2.python import workspace

from core.config import assert_and_infer_cfg
from core.config import cfg
from core.config import get_output_dir
from core.config import merge_cfg_from_file
from core.config import merge_cfg_from_list
from core.test_engine import run_inference
from datasets import task_evaluation
import utils.c2
import utils.logging

utils.c2.import_detectron_ops()
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='optional config file',
        default=None,
        type=str
    )
    parser.add_argument(
        '--wait',
        dest='wait',
        help='wait until net file exists',
        default=True,
        type=bool
    )
    parser.add_argument(
        '--vis', dest='vis', help='visualize detections', action='store_true'
    )
    parser.add_argument(
        '--multi-gpu-testing',
        dest='multi_gpu_testing',
        help='using cfg.NUM_GPUS for inference',
        action='store_true'
    )
    parser.add_argument(
        '--range',
        dest='range',
        help='start (inclusive) and end (exclusive) indices',
        default=None,
        type=int,
        nargs=2
    )
    parser.add_argument(
        '--weights-folder',
        dest='weights_folder',
        help='weights-folder',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output-folder',
        dest='output_folder',
        help='output folder',
        default=None,
        type=str
    )
    parser.add_argument(
        'opts',
        help='See lib/core/config.py for all options',
        default=None,
        nargs=argparse.REMAINDER
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main(ind_range=None, multi_gpu_testing=False):
    output_dir = get_output_dir(training=False)
    all_results = run_inference(
        output_dir, ind_range=ind_range, multi_gpu_testing=multi_gpu_testing
    )
    if not ind_range:
        task_evaluation.check_expected_results(
            all_results,
            atol=cfg.EXPECTED_RESULTS_ATOL,
            rtol=cfg.EXPECTED_RESULTS_RTOL
        )
        import json
        json.dump(all_results, open(os.path.join(output_dir, 'bbox_results_all.json'), 'w'))
        task_evaluation.log_copy_paste_friendly_results(all_results)


def get_list_of_weight_files_todo(weights_folder, output_folder=None):
    import os
    from glob import glob
    weights_files = [y for x in os.walk(weights_folder) for y in glob(os.path.join(x[0], '*.pkl'))]
    list_of_weigths_targets_filtered = []
    for file_path in weights_files:
        iter_number = file_path.rsplit('_', 1)[1][4:-4]
        if not iter_number:
            iter_number = 'final_model'
        result_folder = os.path.join(output_folder, iter_number)
        if os.path.isdir(result_folder) and \
                os.path.isfile(os.path.join(result_folder,
                                            'test/nuclei_stage_1_local_val_split/generalized_rcnn/bbox_results_all.json')):
            continue
        if os.path.isdir(result_folder) and \
                os.path.isfile(os.path.join(result_folder,
                                            'test/nuclei_stage_1_test/generalized_rcnn/bbox_results_all.json')):
            continue
        if os.path.isdir(result_folder) and \
                os.path.isfile(os.path.join(result_folder,
                                            'test/nuclei_stage_2_test/generalized_rcnn/bbox_results_all.json')):
            continue
        list_of_weigths_targets_filtered.append((file_path, result_folder, int(iter_number)))

    list_of_weigths_targets_filtered = sorted(list_of_weigths_targets_filtered, key=lambda item: item[2],  reverse=True)

    return list_of_weigths_targets_filtered


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    logger = utils.logging.setup_logging(__name__)
    args = parse_args()
    logger.info('Called with args:')
    logger.info(args)
    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)
    if args.opts is not None:
        merge_cfg_from_list(args.opts)
    assert_and_infer_cfg()
    logger.info('Testing with config:')
    logger.info(pprint.pformat(cfg))

    if cfg.TEST.WEIGHTS:
        if args.output_folder:
            cfg.OUTPUT_DIR = args.output_folder
        main(ind_range=args.range, multi_gpu_testing=args.multi_gpu_testing)

    else:
        list_of_weigths_targets = get_list_of_weight_files_todo(
            weights_folder=args.weights_folder,
            output_folder=args.output_folder)

        if not list_of_weigths_targets:
            time.sleep(30)

        for weights_file, output_folder, iter_number in list_of_weigths_targets:
            # All arguments to inference functions are passed via cfg
            cfg.TEST.WEIGHTS = weights_file
            cfg.OUTPUT_DIR = output_folder

            # Clear memory before inference
            workspace.ResetWorkspace()

            while not os.path.exists(cfg.TEST.WEIGHTS) and args.wait:
                logger.info('Waiting for \'{}\' to exist...'.format(cfg.TEST.WEIGHTS))
                time.sleep(10)

            main(ind_range=args.range, multi_gpu_testing=args.multi_gpu_testing)
