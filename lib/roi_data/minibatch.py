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
#
# Based on:
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Construct minibatches for Detectron networks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import logging
import numpy as np
import random

from core.config import cfg, get_worker_seed
from core.config import merge_cfg_from_cfg

import roi_data.fast_rcnn
import roi_data.retinanet
import roi_data.rpn
import utils.blob as blob_utils

logger = logging.getLogger(__name__)


def get_minibatch_blob_names(is_training=True):
    """Return blob names in the order in which they are read by the data loader.
    """
    # data blob: holds a batch of N images, each with 3 channels
    blob_names = ['data']
    if cfg.RPN.RPN_ON:
        # RPN-only or end-to-end Faster R-CNN
        blob_names += roi_data.rpn.get_rpn_blob_names(is_training=is_training)
    elif cfg.RETINANET.RETINANET_ON:
        blob_names += roi_data.retinanet.get_retinanet_blob_names(
            is_training=is_training
        )
    else:
        # Fast R-CNN like models trained on precomputed proposals
        blob_names += roi_data.fast_rcnn.get_fast_rcnn_blob_names(
            is_training=is_training
        )
    return blob_names


def get_worker_id():
    import multiprocessing
    process_ident = multiprocessing.current_process()._identity
    worker_id = process_ident[0] if (process_ident and len(process_ident) > 0) else 1
    return worker_id


RAND_LOG_DIR = '/detectron/lib/datasets/data/randlogs/'


def put_minibatch_in_queue(minibatch_queue_mp):
    from utils.coordinator import coordinated_put
    import time, cPickle, os, json
    from utils.coordinator import Coordinator

    with open('/detectron/lib/datasets/data/roidb.pkl') as f:
        roidb = cPickle.load(f)

    with open('/detectron/lib/datasets/data/cfg.pkl') as f:
        other_cfg = cPickle.load(f)
        merge_cfg_from_cfg(other_cfg)

    try:
        os.makedirs(RAND_LOG_DIR)
    except:
        pass
    RAND_SEED = get_worker_seed()

    with open(os.path.join(RAND_LOG_DIR, str(get_worker_id())), 'a') as f:
        json.dump({'RANDOM_SEED': str(RAND_SEED)}, f)
        f.write(os.linesep)

    np.random.seed(RAND_SEED)

    coordinator = Coordinator()

    # filter out really big images
    roidb = [r for r in roidb if len(r['segms']) <= 370]

    perm, cur = _shuffle_roidb_inds(roidb)
    while True:
        t = time.time()

        db_inds, perm, cur = _get_next_minibatch_inds(roidb, perm, cur)
        minibatch_db = [roidb[i] for i in db_inds]
        blobs, valid = get_minibatch(minibatch_db)

        if not valid:
            continue
        # Blobs must be queued in the order specified by
        # self.get_output_names
        from collections import OrderedDict
        ordered_blobs = OrderedDict()
        for key in get_minibatch_blob_names():
            assert blobs[key].dtype in (np.int32, np.float32), \
                'Blob {} of dtype {} must have dtype of ' \
                'np.int32 or np.float32'.format(key, blobs[key].dtype)
            ordered_blobs[key] = blobs[key]
        coordinated_put(
            coordinator, minibatch_queue_mp, ordered_blobs
        )


def _get_next_minibatch_inds(roidb, perm, cur):
    """Return the roidb indices for the next minibatch. Thread safe."""
    # We use a deque and always take the *first* IMS_PER_BATCH items
    # followed by *rotating* the deque so that we see fresh items
    # each time. If the length of _perm is not divisible by
    # IMS_PER_BATCH, then we end up wrapping around the permutation.
    db_inds = [perm[i] for i in range(cfg.TRAIN.IMS_PER_BATCH)]
    perm.rotate(-cfg.TRAIN.IMS_PER_BATCH)
    cur += cfg.TRAIN.IMS_PER_BATCH
    if cur >= len(perm):
        perm, cur = _shuffle_roidb_inds(roidb)
    return db_inds, perm, cur


def _shuffle_roidb_inds(roidb):
    """Randomly permute the training roidb. Not thread safe."""
    if cfg.TRAIN.NORMALIZE_CLASSES:
        # logger.info('========== Normalizing classes ========')
        nuclei_classes = np.array([r['nuclei_class'] for r in roidb])
        groups = [np.where(nuclei_classes == c)[0] for c in set(nuclei_classes)]
        n_per_group = int(len(roidb) / len(groups))

        groups_sampled = [sample_n(g, n_per_group) for g in groups]
        IDXs = np.asarray([item for sublist in groups_sampled for item in sublist])

    else:
        IDXs = np.arange(len(roidb))

    if cfg.TRAIN.ASPECT_GROUPING:
        # logger.info('========== Aspect Grouping ========')
        if len(IDXs) % 2 != 0:
            IDXs = np.concatenate([IDXs, [IDXs[0]]])

        widths = np.array([roidb[r]['width'] for r in IDXs])
        heights = np.array([roidb[r]['height'] for r in IDXs])
        horz = (widths >= heights)
        vert = np.logical_not(horz)
        horz_inds = [IDXs[r] for r in np.where(horz)[0]]
        vert_inds = [IDXs[r] for r in np.where(vert)[0]]
        inds = np.hstack(
            (
                np.random.permutation(horz_inds),
                np.random.permutation(vert_inds)
            )
        )
        inds = np.reshape(inds, (-1, 2))
        row_perm = np.random.permutation(np.arange(inds.shape[0]))
        inds = np.reshape(inds[row_perm, :], (-1, ))
        perm = inds
    else:
        perm = np.random.permutation(IDXs)
    from collections import deque
    perm = deque(np.int32(perm))
    cur = 0

    import os, json
    with open(os.path.join(RAND_LOG_DIR, str(get_worker_id())), 'a') as f:
        json.dump({'PERMUTATION': str(perm)}, f)
        f.write(os.linesep)

    return perm, cur


def sample_n(Idxs, n):
    RIDxs = []
    while len(RIDxs) < n:
        rem = n - len(RIDxs)
        if len(Idxs) <= rem:
            RIDxs.extend(Idxs)
        else:
            P_Idxs = np.random.permutation(Idxs)
            RIDxs.extend(P_Idxs[:rem])
    return RIDxs  # numpy.random.permutation(RIDxs)


def get_minibatch(roidb):
    """Given a roidb, construct a minibatch sampled from it."""
    from datasets.nuclei.augmentation import augment_images

    if cfg.TRAIN.USE_AUGMENTOR:
        roidb, augmented_ims = augment_images(roidb, cfg.TRAIN.USE_MASK_AUGMENTOR, cfg.TRAIN.USE_BBOX_AUGMENTOR)
        im_blob, im_scales = _get_image_blob_from_images(roidb, augmented_ims)
    else:
        im_blob, im_scales = _get_image_blob(roidb)

    # We collect blobs from each image onto a list and then concat them into a
    # single tensor, hence we initialize each blob to an empty list
    blobs = {k: [] for k in get_minibatch_blob_names()}
    # Get the input image blob, formatted for caffe2
    blobs['data'] = im_blob
    if cfg.RPN.RPN_ON:
        # RPN-only or end-to-end Faster/Mask R-CNN
        valid = roi_data.rpn.add_rpn_blobs(blobs, im_scales, roidb)
    elif cfg.RETINANET.RETINANET_ON:
        im_width, im_height = im_blob.shape[3], im_blob.shape[2]
        # im_width, im_height corresponds to the network input: padded image
        # (if needed) width and height. We pass it as input and slice the data
        # accordingly so that we don't need to use SampleAsOp
        valid = roi_data.retinanet.add_retinanet_blobs(
            blobs, im_scales, roidb, im_width, im_height
        )
    else:
        # Fast R-CNN like models trained on precomputed proposals
        valid = roi_data.fast_rcnn.add_fast_rcnn_blobs(blobs, im_scales, roidb)
    return blobs, valid


def _get_image_blob_from_images(roidb, images):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    scale_inds = np.random.randint(
        0, high=len(cfg.TRAIN.SCALES), size=num_images
    )
    processed_ims = []
    im_scales = []
    for i in range(num_images):
        im = images[i]
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]

        if cfg.TRAIN.USE_INVERSE and random.choice([True, False]):
            im = 255 - im

        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        im, im_scale = blob_utils.prep_im_for_blob(
            im, cfg.PIXEL_MEANS, [target_size], cfg.TRAIN.MAX_SIZE
        )
        im_scales.append(im_scale[0])
        processed_ims.append(im[0])

    # Create a blob to hold the input images
    blob = blob_utils.im_list_to_blob(processed_ims)

    return blob, im_scales


def _get_image_blob(roidb):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    scale_inds = np.random.randint(
        0, high=len(cfg.TRAIN.SCALES), size=num_images
    )
    processed_ims = []
    im_scales = []
    for i in range(num_images):
        im = cv2.imread(roidb[i]['image'])
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]

        if cfg.TRAIN.USE_INVERSE and random.choice([True, False]):
            im = 255 - im

        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        im, im_scale = blob_utils.prep_im_for_blob(
            im, cfg.PIXEL_MEANS, [target_size], cfg.TRAIN.MAX_SIZE
        )
        im_scales.append(im_scale[0])
        processed_ims.append(im[0])

    # Create a blob to hold the input images
    blob = blob_utils.im_list_to_blob(processed_ims)

    return blob, im_scales
