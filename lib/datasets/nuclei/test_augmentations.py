from datasets.nuclei.nuclei_utils import rescale_to_255, show
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import cv2
from datasets.json_dataset import JsonDataset
from pycocotools import mask as mask_util


def load_all_roidb():
    roidb = []

    ds = ('nuclei_stage_1_local_train_split',
          'nuclei_stage_1_local_val_split',
          # 'nuclei_stage_1_test',
          'nucleisegmentationbenchmark',
          'cluster_nuclei',
          'BBBC006',
          'BBBC007',
          'BBBC018',
          'BBBC020',
          '2009_ISBI_2DNuclei',
          # 'nuclei_partial_annotations',
          'TNBC_NucleiSegmentation',
          )

    for d in ds:
        roidb.extend(JsonDataset(d).get_roidb(gt=True))

    return roidb


def augment_grid_image():
    image_id = '4d09672bcf5a2661eea00891bbb8191225a06619a849aece37ad10d9dedbde3e.jpg'
    image_id = '1b44d22643830cd4f23c9deadb0bd499fb392fb2cd9526d81547d93077d983df.jpg'
    roidb = load_all_roidb()
    for roi in roidb:
        if roi['image'].rsplit('/', 1)[1] == image_id:
            break

    im = cv2.imread(roi['image'])[:, :, [2, 1, 0]]

    M = mask_util.decode(roi['segms'])

    from datasets.nuclei.augmentation import seq_optimistic, hooks_optimistic_masks, seq_old, hooks_masks_old

    seq_det = seq_old.to_deterministic()
    seq_det.show_grid(im, cols=9, rows=9)


def load_roidb():
    dataset_name = 'nuclei_stage_1_local_train_split'
    ds = JsonDataset(dataset_name)
    roidb = ds.get_roidb(gt=True)
    return roidb


def get_images_masks(roidb):
    import cv2
    ims = [cv2.imread(roi['image'])[:, :, [2, 1, 0]] for roi in roidb]
    orig_masks = [mask_util.decode(roi['segms']) for roi in roidb]

    return ims, orig_masks


roidb = load_all_roidb()

cluster_file_path = 'clusters.json'

samples = {}
for r in roidb:
    if np.random.random() > 0.5:
        samples[r['nuclei_class']] = r

from core.config import cfg
cfg.TRAIN.USE_AUGMENTOR = True
cfg.TRAIN.AUGMENTATION_MODE = 'SIMPLE'
cfg.TRAIN.AUGMENTATION_MODE = 'COMPREHENSIVE'
cfg.TRAIN.AUGMENTATION_MODE = 'OPTIMISTIC'
cfg.GRAY_IMAGES = True

from collections import Counter

counts = Counter()
for r in roidb:
    counts[r['nuclei_class']] += 1


classes = set()
for r in roidb:
    classes.add(r['nuclei_class'])


for i in classes:

    samples = []
    for r in roidb:
        if r['nuclei_class'] == i and np.random.random() > 0.5:
            samples.append(r)
        if len(samples) == 5:
            break

    print i, len(samples)

    for roi in samples:
        roi['bbox_targets'] = []

    IMS = []
    AUG_ROIDB = []
    for r in samples:
        from datasets.nuclei.augmentation import augment_images

        roi_augs, im_augs = augment_images([r])

        IMS.append(im_augs[0])
        AUG_ROIDB.append(roi_augs[0])
    for I, r in zip(IMS, AUG_ROIDB):
        show(I.copy())
        M = mask_util.decode(r['segms'])
        show(np.sum(M, -1))

    import time
    time.sleep(20)


im_roi = {}
for roi in roidb:
    im_roi[roi['image'].rsplit('/',1)[1]] = roi

roi = im_roi['19467-DNA.DIB.jpg']
roi['bbox_targets'] = []

for i in range(10):
    res = load_and_preprocess([roi], train=True)
    M = mask_util.decode(res[0][0]['segms'])
    show(np.sum(M, -1))


for i in range(10):

    samples = {}
    for r in roidb:
        if np.random.random() > 0.5:
            samples[r['nuclei_class']] = r

    for roi in samples.values():
        roi['bbox_targets'] = []

    IMS = []
    for r in samples.values():
        try:
            IMS.append(load_and_preprocess([r], train=True))
        except Exception as e:
            print e

    for r, I in IMS:
        show(hsv_scaled_to_rgb_256(I[0].copy()))

    import time
    time.sleep(10)


samples = {}
for r in roidb:
    if np.random.random() > 0.5:
        samples[r['nuclei_class']] = r

for roi in samples.values():
    roi['bbox_targets'] = []

IMS = []
for r in samples.values():
    try:
        IMS.append(load_and_preprocess([r], train=True))
    except Exception as e:
        print e

for r, I in IMS:
    show(hsv_scaled_to_rgb_256(I[0].copy()))


for r, I in IMS:
    show(I[0][:, :, 2].copy())


ims, M = get_images_masks(samples.values())
from datasets.nuclei.augmentation import get_img_aug
seq, hooks_masks = get_img_aug()

mask_augs = []
ims_augs = []
for im, m in zip(ims,M):
    seq_det = seq.to_deterministic()
    maug = seq_det.augment_image(m, hooks=hooks_masks)
    mask_augs.append(maug)
    show(np.sum(m, -1))
    show(np.sum(maug, -1))
    imaug = seq_det.augment_image(im)
    ims_augs.append(imaug)
    show(imaug)


for im in ims:
    seq_det = seq.to_deterministic()
    seq_det.show_grid(im, cols=5, rows=5)


def load_samples_per_class():
    roidb = load_all_roidb()

    cluster_file_path = 'clusters.json'

    import json, os
    Clusters = json.load(open(cluster_file_path))

    for roi in roidb:
        im_id = roi['image'].rsplit('/', 1)[1]
        if im_id in Clusters:
            roi['nuclei_class'] = Clusters[im_id]
        elif im_id.split(':', 1)[0] in Clusters:
            roi['nuclei_class'] = Clusters[im_id.split(':', 1)[0]]
        else:
            roi['nuclei_class'] = '-1'

    samples = {}
    for r in roidb:
        if np.random.random() > 0.5:
            samples[r['nuclei_class']] = r

    from utils.image import load_and_preprocess, hsv_scaled_to_rgb_256
    from core.config import cfg
    cfg.TRAIN.USE_AUGMENTOR = True
    cfg.TRAIN.AUGMENTATION_MODE = 'SIMPLE'

    for roi in samples.values():
        roi['bbox_targets'] = []

    IMS = []
    for r in samples.values():
        IMS.append(load_and_preprocess([r], train=True))

    for r, I in IMS:
        import imageio
        show(imageio.imread(r[0]['image']))
        show(hsv_scaled_to_rgb_256(I[0]))

    ims, M = get_images_masks(samples.values())

    from datasets.nuclei.augmentation import black_and_white_aug, show_masks
    seq, hooks_masks = black_and_white_aug()

    seq_det = seq.to_deterministic()

    for im in ims:
        seq_det.show_grid(rescale_to_255(np.asarray(im, dtype=np.float32)), cols=10, rows=10)


roidb = load_roidb()
ims, orig_masks = get_images_masks(roidb[:10])

from datasets.nuclei.augmentation import black_and_white_aug, scale_boxes

seq, hooks_masks = black_and_white_aug()
seq_det = seq.to_deterministic()
seq_det.show_grid(rescale_to_255(np.asarray(ims[1], dtype=np.float32)), cols=10, rows=10)

for i in range(10):
    seq_det = seq.to_deterministic()
    seq_det.show_grid(rescale_to_255(np.asarray(ims[1], dtype=np.float32)), cols=10, rows=10)

import time
start = time.time()
aug_ims_augs = seq_det.augment_images(ims)
print time.time() - start

start = time.time()
mask_augs = seq_det.augment_images(orig_masks, hooks=hooks_masks)
print time.time() - start



def masks_to_labels(masks):
    # Make a ground truth label image (pixel value is index of object label)
    labels = np.zeros(masks.shape[:2], np.float64)
    for index in range(0, masks.shape[-1]):
        labels[masks[:, :, index] > 0] = 100 * index + 1

    return labels


def labels_to_masks(labels, sz):
    # Make a ground truth label image (pixel value is index of object label)
    masks = np.zeros((labels.shape[0], labels.shape[1], sz), np.uint16)
    for index in range(0, masks.shape[-1]):
        masks[:, :, index] = (labels > 100 * index + 0.5) & (labels < 100 * index + 1.5)

    return masks

start = time.time()
label_masks = [masks_to_labels(M) for M in orig_masks]
sizes = [M.shape[-1] for M in orig_masks]
mask_augs = seq_det.augment_images(label_masks, hooks=hooks_masks)
aug_masks = [labels_to_masks(l,sz) for l, sz in zip(mask_augs, sizes)]

print time.time() - start




for M in mask_augs:
    M[M > 124] = 255
    M[M <= 124] = 0

import time
time.sleep(2)
for im, mask in zip(ims, orig_masks):
    show(im)
    show(np.sum(mask, -1))

import time
time.sleep(2)
for im, mask in zip(aug_ims_augs, mask_augs):
    show(im)
    show(np.sum(mask, -1) > 0)

from pycocotools import mask as mask_util

aug_rles = mask_util.encode(np.asarray(orig_masks[0], order='F'))
boxes = np.asarray(mask_util.toBbox(aug_rles))
b = np.int32(scale_boxes(boxes.copy()))


def save_im_masks(im, M, id, dir):
    from utils.boxes import xywh_to_xyxy
    import os
    try:
        os.mkdir(os.path.join('vis', dir))
    except:
        pass
    M[M > 0] = 1
    aug_rles = mask_util.encode(np.asarray(M, order='F'))
    boxes = xywh_to_xyxy(np.asarray(mask_util.toBbox(aug_rles)))
    boxes = np.append(boxes, np.ones((len(boxes), 2)), 1)

    from utils.vis import vis_one_image
    vis_one_image(im, str(id), os.path.join('vis', dir), boxes, segms=aug_rles, keypoints=None, thresh=0.9,box_alpha=0.8,  show_class=False,)


for idx, im, mask in zip(range(len(aug_ims_augs)), aug_ims_augs, mask_augs):
    save_im_masks(im, mask, idx, 'aug')

for idx, im, mask in zip(range(len(ims)), ims, orig_masks):
    save_im_masks(im, mask, idx, 'orig')

for i in range(5):
    seq_det = seq.to_deterministic()
    seq_det.show_grid(rescale_to_255(np.asarray(ims[1], dtype=np.float32)), cols=10, rows=10)


for M in orig_masks:
    M[M > 0] = 255

seq_det.show_grid(np.sum(orig_masks[1], -1), cols=10, rows=10)


show(ims[1])
show(np.sum(orig_masks[0], -1))

show((np.sum(mask_augs[0], -1) == 255) ^ (np.sum(mask_augs[0], -1) > 0))

for idx in range(len(mask_augs)):
    show((np.sum(mask_augs[idx], -1) == 255) ^ (np.sum(mask_augs[idx], -1) > 0))


for roi in roidb:
    if im_name == roi['image']:
        break

from pycocotools import mask as mask_util
M = mask_util.decode(roi['segms'])
print M.shape
print len(roi['segms'])

import json

D = json.load(open('data/stage1_fixed.json'))


for im in D['images']:
    if im['file_name'] == u'20b20ab049372d184c705acebe7af026d3580f5fd5a72ed796e3622e1685af2f.jpg':
        print im['id']
        break
A = []
for ann in D['annotations']:
    if ann['image_id'] == 87:
        A.append(ann)

print len(A)