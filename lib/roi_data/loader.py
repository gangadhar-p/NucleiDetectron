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

"""Detectron data loader. The design is generic and abstracted away from any
details of the minibatch. A minibatch is a dictionary of blob name keys and
their associated numpy (float32 or int32) ndarray values.

Outline of the data loader design:

loader thread\
loader thread \                    / GPU 1 enqueue thread -> feed -> EnqueueOp
...           -> minibatch queue ->  ...
loader thread /                    \ GPU N enqueue thread -> feed -> EnqueueOp
loader thread/

<---------------------------- CPU -----------------------------|---- GPU ---->

A pool of loader threads construct minibatches that are put onto the shared
minibatch queue. Each GPU has an enqueue thread that pulls a minibatch off the
minibatch queue, feeds the minibatch blobs into the workspace, and then runs
an EnqueueBlobsOp to place the minibatch blobs into the GPU's blobs queue.
During each fprop the first thing the network does is run a DequeueBlobsOp
in order to populate the workspace with the blobs from a queued minibatch.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import deque
from collections import OrderedDict
import logging
from multiprocessing import Pool
from multiprocessing import Queue as MPQueue

import numpy as np
import Queue
import signal
import threading
import time
import uuid
import copy

from caffe2.python import core, workspace

from core.config import cfg
from roi_data.minibatch import get_minibatch
from roi_data.minibatch import get_minibatch_blob_names
from utils.coordinator import coordinated_get
from utils.coordinator import coordinated_put
from utils.coordinator import Coordinator
import utils.c2 as c2_utils

logger = logging.getLogger(__name__)


class RoIDataLoader(object):
    def __init__(
        self,
        roidb,
        num_loaders=4,
        minibatch_queue_size=64,
        blobs_queue_capacity=8,
        num_augmentation_processes=8,
    ):
        self._roidb = roidb
        for roi in roidb:
            roi.pop('dataset', None)  # pop the reference to prevent duplication
        self._lock = threading.Lock()
        self._perm = deque(range(len(self._roidb)))
        self._cur = 0  # _perm cursor
        self._counter = 0
        # The minibatch queue holds prepared training data in host (CPU) memory
        # When training with N > 1 GPUs, each element in the minibatch queue
        # is actually a partial minibatch which contributes 1 / N of the
        # examples to the overall minibatch
        self._minibatch_queue = Queue.Queue(maxsize=minibatch_queue_size)
        self._minibatch_queue_mp = MPQueue(minibatch_queue_size)

        self._blobs_queue_capacity = blobs_queue_capacity
        # Random queue name in case one instantiates multple RoIDataLoaders
        self._loader_id = uuid.uuid4()
        self._blobs_queue_name = 'roi_blobs_queue_{}'.format(self._loader_id)
        # Loader threads construct (partial) minibatches and put them on the
        # minibatch queue
        self._num_loaders = num_loaders
        self._num_gpus = cfg.NUM_GPUS
        self.coordinator = Coordinator()

        self._output_names = get_minibatch_blob_names()
        self._shuffle_roidb_inds()
        self.create_threads()
        self.num_augmentation_processes = num_augmentation_processes

    def minibatch_loader_thread(self):
        """Load mini-batches and put them onto the mini-batch queue."""
        augmentation_process_pool = None
        mini_thread_batch_iter = 0
        with self.coordinator.stop_on_exception():
            while not self.coordinator.should_stop():
                t = time.time()

                if mini_thread_batch_iter % 10 == 0:
                    if augmentation_process_pool:
                        augmentation_process_pool.close()
                        augmentation_process_pool.join()
                        logger.info('get_next_parallel_minibatch DELETE POOL Thread: {} took time: {} MINI_ITER: {}'.format(threading.currentThread(), time.time() - t, mini_thread_batch_iter))

                    augmentation_process_pool = self.crate_augmentation_process_pool(self.num_augmentation_processes)
                    logger.info('get_next_parallel_minibatch CREATE POOL Thread: {} took time: {} MINI_ITER: {}'.format(threading.currentThread(), time.time() - t, mini_thread_batch_iter))

                t = time.time()
                logger.info('get_next_parallel_minibatch: Going to prepare for thread: {} MINI_ITER: {}'.format(threading.currentThread(), mini_thread_batch_iter))
                blobs_list = self.get_next_parallel_minibatch(augmentation_process_pool, self.num_augmentation_processes)
                logger.info('get_next_parallel_minibatch Thread: {}  {}: len of blobs_list: {} MINI_ITER: {}'.format(threading.currentThread(), time.time() - t, len(blobs_list), mini_thread_batch_iter))
                t = time.time()
                for blobs in blobs_list:
                    # Blobs must be queued in the order specified by
                    # self.get_output_names
                    ordered_blobs = OrderedDict()
                    for key in self.get_output_names():
                        assert blobs[key].dtype in (np.int32, np.float32), \
                            'Blob {} of dtype {} must have dtype of ' \
                            'np.int32 or np.float32'.format(key, blobs[key].dtype)
                        ordered_blobs[key] = blobs[key]
                    coordinated_put(
                        self.coordinator, self._minibatch_queue, ordered_blobs
                    )

                logger.debug('coordinated_put {}: len of blobs_list: {} MINI_ITER: {}'.format(time.time() - t, len(blobs_list), mini_thread_batch_iter))

                t = time.time()
                # del blobs_list
                mini_thread_batch_iter += 1

        if augmentation_process_pool:
            augmentation_process_pool.close()
            augmentation_process_pool.join()
            logger.info('get_next_parallel_minibatch DELETE POOL Thread: {}'.format(threading.currentThread()))

        logger.info('Stopping mini-batch loading thread')

    def rescale_0_1(self, X):
        # X = np.random.uniform(low=-1, high=1, size=(100,120))
        from sklearn.preprocessing import MinMaxScaler
        min_max_scaler = MinMaxScaler(feature_range=(0, 1), copy=False)
        sz = X.shape
        X = min_max_scaler.fit_transform(X.reshape(-1,1))
        X = X.reshape(sz)
        return X

    def save_im_masks(self, blobs):
        import os, uuid
        from datasets.dataset_catalog import _DATA_DIR
        import utils.blob as blob_utils

        channel_swap = (0, 2, 3, 1)
        data = blobs['data'].copy()

        im = data.transpose(channel_swap)[0]
        im = self.rescale_0_1(im)

        roidb_temp = blob_utils.deserialize(blobs['roidb'])[0]

        im_name = str(self._counter) + '_' + os.path.splitext(os.path.basename(roidb_temp['image']))[0]

        with self._lock:
            self._counter += 1

        out_dir = os.path.join(_DATA_DIR, 'vis', roidb_temp['nuclei_class'])
        im_name += '_' + str(uuid.uuid4().get_hex().upper()[0:6])

        try:
            os.makedirs(out_dir)
        except:
            pass

        aug_rles = roidb_temp['segms']

        boxes = roidb_temp['boxes']
        boxes = np.append(boxes, np.ones((len(boxes), 2)), 1)
        im_scale = blobs['im_info'][0, 2]

        from utils.vis import vis_one_image
        vis_one_image(im, im_name, out_dir, boxes, segms=aug_rles, keypoints=None, thresh=0.7,
                      box_alpha=0.8, show_class=False, scale=im_scale)

    def enqueue_blobs_thread_mp(self, gpu_id, blob_names):
        """Transfer mini-batches from a mini-batch queue to a BlobsQueue."""
        with self.coordinator.stop_on_exception():
            while not self.coordinator.should_stop():
                if self._minibatch_queue_mp.qsize() == 0:
                    logger.warning('minibatch_queue_mp -batch queue is empty')
                blobs = coordinated_get(self.coordinator, self._minibatch_queue_mp)

                if cfg.LOG_IMAGES and np.random.random() <= 0.01:
                    self.save_im_masks(blobs)

                self.enqueue_blobs(gpu_id, blob_names, blobs.values())
                logger.debug(
                    'batch queue size {}'.format(self._minibatch_queue_mp.qsize())
                )
            logger.info('Stopping enqueue thread')

    def enqueue_blobs_thread(self, gpu_id, blob_names):
        """Transfer mini-batches from a mini-batch queue to a BlobsQueue."""
        with self.coordinator.stop_on_exception():
            while not self.coordinator.should_stop():
                if self._minibatch_queue.qsize == 0:
                    logger.warning('Mini-batch queue is empty')
                blobs = coordinated_get(self.coordinator, self._minibatch_queue)
                self.enqueue_blobs(gpu_id, blob_names, blobs.values())
                logger.info(
                    'batch queue size {}'.format(self._minibatch_queue.qsize())
                )
            logger.info('Stopping enqueue thread')

    def get_next_parallel_minibatch(self, augmentation_process_pool, parallel_size):
        """Return the blobs to be used for the next minibatch. Thread safe."""

        db_inds_list = [self._get_next_minibatch_inds() for _ in range(parallel_size)]
        minibatch_db_list = [[self._roidb[i] for i in db_inds] for db_inds in db_inds_list]

        copy_minibatch_db_list = copy.deepcopy(minibatch_db_list)

        logger.info('get_next_parallel_minibatch MAP POOL Thread: {}'.format(threading.currentThread()))
        res = augmentation_process_pool.map(get_minibatch, copy_minibatch_db_list)
        logger.info('get_next_parallel_minibatch MAP POOL COMPLETE Thread: {}'.format(threading.currentThread()))

        valid_blobs_list = [blobs for blobs, valid in res if valid]

        del copy_minibatch_db_list
        del minibatch_db_list
        del res

        return valid_blobs_list

    def get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch. Thread safe."""
        valid = False
        while not valid:
            db_inds = self._get_next_minibatch_inds()
            minibatch_db = [self._roidb[i] for i in db_inds]
            blobs, valid = get_minibatch(minibatch_db)
        return blobs

    def sample_n(self, Idxs, n):
        RIDxs = []
        while len(RIDxs) < n:
            rem = n - len(RIDxs)
            if len(Idxs) <= rem:
                RIDxs.extend(Idxs)
            else:
                P_Idxs = np.random.permutation(Idxs)
                RIDxs.extend(P_Idxs[:rem])
        return RIDxs  # numpy.random.permutation(RIDxs)

    def _shuffle_roidb_inds(self):

        """Randomly permute the training roidb. Not thread safe."""
        if cfg.TRAIN.NORMALIZE_CLASSES:
            # logger.info('========== Normalizing classes ========')
            nuclei_classes = np.array([r['nuclei_class'] for r in self._roidb])
            groups = [np.where(nuclei_classes == c)[0] for c in set(nuclei_classes)]
            n_per_group = int(len(self._roidb) / len(groups))

            groups_sampled = [self.sample_n(g, n_per_group) for g in groups]
            IDXs = np.asarray([item for sublist in groups_sampled for item in sublist])

        else:
            IDXs = np.arange(len(self._roidb))

        if cfg.TRAIN.ASPECT_GROUPING:
            # logger.info('========== Aspect Grouping ========')
            # THIS ONLY HAPPENS WHEN DATASET SIZE IS NOT EVEN NUMBER AT LINE NO: 311
            # ADDING TESTING STAGE 1 DATA TRIGGERED THIS BUG AND COULD NOT FINETUNE

            if len(IDXs) % 2 != 0:
                IDXs = np.concatenate([IDXs, [IDXs[0]]])

            widths = np.array([self._roidb[r]['width'] for r in IDXs])
            heights = np.array([self._roidb[r]['height'] for r in IDXs])
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
            self._perm = inds
            # logger.info('Permutations: {}'.format(inds))
        else:
            self._perm = np.random.permutation(IDXs)
        self._perm = deque(np.int32(self._perm))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch. Thread safe."""
        with self._lock:
            # We use a deque and always take the *first* IMS_PER_BATCH items
            # followed by *rotating* the deque so that we see fresh items
            # each time. If the length of _perm is not divisible by
            # IMS_PER_BATCH, then we end up wrapping around the permutation.
            db_inds = [self._perm[i] for i in range(cfg.TRAIN.IMS_PER_BATCH)]
            self._perm.rotate(-cfg.TRAIN.IMS_PER_BATCH)
            self._cur += cfg.TRAIN.IMS_PER_BATCH
            if self._cur >= len(self._perm):
                self._shuffle_roidb_inds()
        return db_inds

    def get_output_names(self):
        return self._output_names

    def enqueue_blobs(self, gpu_id, blob_names, blobs):
        """Put a mini-batch on a BlobsQueue."""
        assert len(blob_names) == len(blobs)
        t = time.time()
        dev = c2_utils.CudaDevice(gpu_id)
        queue_name = 'gpu_{}/{}'.format(gpu_id, self._blobs_queue_name)
        blob_names = ['gpu_{}/{}'.format(gpu_id, b) for b in blob_names]
        for (blob_name, blob) in zip(blob_names, blobs):
            workspace.FeedBlob(blob_name, blob, device_option=dev)
        logger.debug(
            'enqueue_blobs {}: workspace.FeedBlob: {}'.
            format(gpu_id, time.time() - t)
        )
        t = time.time()
        op = core.CreateOperator(
            'SafeEnqueueBlobs', [queue_name] + blob_names,
            blob_names + [queue_name + '_enqueue_status'],
            device_option=dev
        )
        workspace.RunOperatorOnce(op)
        logger.debug(
            'enqueue_blobs {}: workspace.RunOperatorOnce: {}'.
            format(gpu_id, time.time() - t)
        )

    def create_threads_and_pool_mp(self):

        # Create one BlobsQueue per GPU
        # (enqueue_blob_names are unscoped)
        enqueue_blob_names = self.create_blobs_queues()

        # Create one enqueuer thread per GPU
        self._enqueuers = [
            threading.Thread(
                target=self.enqueue_blobs_thread_mp,
                args=(gpu_id, enqueue_blob_names)
            ) for gpu_id in range(self._num_gpus)
        ]

    def create_threads(self):
        if cfg.DATA_LOADER.PROCESS_POOL_LOADER:
            self.create_threads_and_pool_mp()
            return

        # Create mini-batch loader threads, each of which builds mini-batches
        # and places them into a queue in CPU memory
        self._workers = [
            threading.Thread(target=self.minibatch_loader_thread)
            for _ in range(self._num_loaders)
        ]

        # Create one BlobsQueue per GPU
        # (enqueue_blob_names are unscoped)
        enqueue_blob_names = self.create_blobs_queues()

        # Create one enqueuer thread per GPU
        self._enqueuers = [
            threading.Thread(
                target=self.enqueue_blobs_thread,
                args=(gpu_id, enqueue_blob_names)
            ) for gpu_id in range(self._num_gpus)
        ]

    def start_mp(self, prefill=False):
        from roi_data.minibatch import put_minibatch_in_queue

        import cPickle
        with open('/detectron/lib/datasets/data/roidb.pkl', 'w') as f:
            cPickle.dump(self._roidb, f)

        with open('/detectron/lib/datasets/data/cfg.pkl', 'w') as f:
            cPickle.dump(cfg, f)

        self.loader_pool = Pool(processes=self.num_augmentation_processes,
                                initializer=put_minibatch_in_queue,
                                initargs=(self._minibatch_queue_mp,))

        for w in self._enqueuers:
            w.start()

        if prefill:
            logger.info('Pre-filling mini-batch queue...')
            while not self._minibatch_queue_mp.full():
                logger.info(
                    '  [{:d}/{:d}]'.format(
                        self._minibatch_queue_mp.qsize(),
                        self._minibatch_queue_mp._maxsize
                    )
                )
                time.sleep(0.1)
                # Detect failure and shutdown
                if self.coordinator.should_stop():
                    self.shutdown()
                    break

    def start(self, prefill=False):
        if cfg.DATA_LOADER.PROCESS_POOL_LOADER:
            self.start_mp(prefill)
            return

        for w in self._workers + self._enqueuers:
            w.start()
        if prefill:
            logger.info('Pre-filling mini-batch queue...')
            while not self._minibatch_queue.full():
                logger.info(
                    '  [{:d}/{:d}]'.format(
                        self._minibatch_queue.qsize(),
                        self._minibatch_queue.maxsize
                    )
                )
                time.sleep(0.1)
                # Detect failure and shutdown
                if self.coordinator.should_stop():
                    self.shutdown()
                    break

    def shutdown_mp(self):
        self.coordinator.request_stop()
        self.coordinator.wait_for_stop()
        self.close_blobs_queues()
        self.loader_pool.terminate()
        self.loader_pool.join()

        for w in self._enqueuers:
            w.join()

    def shutdown(self):
        if cfg.DATA_LOADER.PROCESS_POOL_LOADER:
            self.shutdown_mp()
            return

        self.coordinator.request_stop()
        self.coordinator.wait_for_stop()
        self.close_blobs_queues()
        for w in self._workers + self._enqueuers:
            w.join()

    def create_blobs_queues(self):
        """Create one BlobsQueue for each GPU to hold mini-batches."""
        for gpu_id in range(self._num_gpus):
            with c2_utils.GpuNameScope(gpu_id):
                workspace.RunOperatorOnce(
                    core.CreateOperator(
                        'CreateBlobsQueue', [], [self._blobs_queue_name],
                        num_blobs=len(self.get_output_names()),
                        capacity=self._blobs_queue_capacity
                    )
                )
        return self.create_enqueue_blobs()

    def close_blobs_queues(self):
        """Close a BlobsQueue."""
        for gpu_id in range(self._num_gpus):
            with core.NameScope('gpu_{}'.format(gpu_id)):
                workspace.RunOperatorOnce(
                    core.CreateOperator(
                        'CloseBlobsQueue', [self._blobs_queue_name], []
                    )
                )

    def create_enqueue_blobs(self):
        blob_names = self.get_output_names()
        enqueue_blob_names = [
            '{}_enqueue_{}'.format(b, self._loader_id) for b in blob_names
        ]
        for gpu_id in range(self._num_gpus):
            with c2_utils.NamedCudaScope(gpu_id):
                for blob in enqueue_blob_names:
                    workspace.CreateBlob(core.ScopedName(blob))
        return enqueue_blob_names

    def register_sigint_handler(self):
        def signal_handler(signal, frame):
            logger.info(
                'SIGINT: Shutting down RoIDataLoader threads and exiting...'
            )
            self.shutdown()

        signal.signal(signal.SIGINT, signal_handler)

    def crate_augmentation_process_pool(self, num_processes):
        pool = Pool(processes=num_processes)
        return pool
