import cv2 as cv
from skimage.morphology import watershed
import numpy as np
from skimage.feature import peak_local_max
from skimage import measure
from skimage.segmentation import random_walker
from scipy import ndimage
from pycocotools import mask as mask_util
import os
from datasets.nuclei.nuclei_utils import save
from datasets.nuclei.nuclei_utils import show

k_3x3 = np.ones((3, 3), np.uint8)
kernels = [
    np.ones((1, 2), np.uint8),
    np.ones((1, 3), np.uint8),
    np.ones((2, 1), np.uint8),
    np.ones((2, 2), np.uint8),
    np.ones((2, 3), np.uint8),
    np.ones((3, 1), np.uint8),
    np.ones((3, 2), np.uint8),
    k_3x3,
]


def fill_holes_in_mask(mask):
    return ndimage.morphology.binary_fill_holes(mask).astype(np.uint8)


def opencv_controur_smooth(mask, kernel, k):
    if len(mask.shape) == 3:
        mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)

    mask = cv.dilate(mask, kernel, iterations=k)

    _, C, h = cv.findContours(mask.copy(), cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
    seg = [[float(x) for x in contour.flatten()] for contour in C]
    seg = [cont for cont in seg if len(cont) > 4]  # filter all polygons that are boxes
    rles = mask_util.frPyObjects(seg, mask.shape[0], mask.shape[1])
    masks = mask_util.decode(rles)
    res = []
    for idx in range(masks.shape[2]):
        res.append(masks[:, :, idx].astype(np.uint8))
    return res


def postprocess_mask(mask, skip=True):
    mask = fill_holes_in_mask(mask)
    if skip:
        return mask
    masks = opencv_controur_smooth(mask, kernels[0], 1)
    masks = [m for m in masks if m.sum() > 30]
    if len(masks) == 0:
        return mask
    if len(masks) == 1:
        return masks[0]
    else:
        areas = [m.sum() for m in masks]
        M = masks[areas.index(max(areas))]
        return M


def split_mask_erode_dilate(mask, kernel=k_3x3, k=3):
    img_erosion = cv.erode(mask, kernel, iterations=k)
    output = cv.connectedComponentsWithStats(img_erosion, 4, cv.CV_32S)
    if output[0] < 2:
        return [mask], output[1]
    else:
        masks_res = []
        for idx in range(1, output[0]):
            res_m = (output[1] == idx).astype(np.uint8)
            res_m = cv.dilate(res_m, kernel, iterations=k)
            if res_m.sum() > 5:
                masks_res.append(res_m)
        return masks_res, output[1]


def opencv_segmentation(mask, kernel=k_3x3, k=3):
    # noise removal
    opening = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=k)

    # sure background area
    sure_bg = cv.dilate(opening, kernel, iterations=k)

    # Finding sure foreground area
    dist_transform = cv.distanceTransform(opening,cv.DIST_L2, 5)
    ret, sure_fg = cv.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown > 0] = 0

    labels_ws = cv.watershed(cv.cvtColor(mask, cv.COLOR_GRAY2RGB), markers)

    if labels_ws.max() - 1 < 2:
        return [mask], labels_ws

    res_masks = []
    for idx in range(2,  labels_ws.max() + 1):
        m = labels_ws == idx
        if m.sum() > 5:
            m = cv.dilate(m.astype(np.uint8), kernel, iterations=1)
            res_masks.append(m)
    return res_masks, labels_ws


def skimage_watershed_segmentation(mask, kernel=k_3x3, k=1):
    # mask = cv.dilate(mask, kernel, iterations=k)

    distance = ndimage.distance_transform_edt(mask)
    local_maxi = peak_local_max(distance, indices=False, footprint=kernel, labels=mask)

    markers = measure.label(local_maxi)
    labels_ws = watershed(-distance, markers, mask=mask)

    if labels_ws.max() < 2:
        return [mask], labels_ws

    res_masks = []
    for idx in range(1,  labels_ws.max() + 1):
        m = labels_ws == idx
        if m.sum() > 20:
            res_masks.append(m.astype(np.uint8))
    return res_masks, labels_ws


def skimage_random_walker_segmentation(mask, kernel=k_3x3, k=1):
    if mask.dtype != np.bool:
        mask = mask > 0

    distance = ndimage.distance_transform_edt(mask)
    local_maxi = peak_local_max(distance, indices=False, footprint=kernel, labels=mask)

    markers = measure.label(local_maxi)
    markers[~mask] = -1
    labels_rw = random_walker(mask, markers)

    if labels_rw.max() < 2:
        return [mask.astype(np.uint8)], labels_rw

    res_masks = []
    for idx in range(1,  labels_rw.max() + 1):
        m = labels_rw == idx
        if m.sum() > 20:
            res_masks.append(m.astype(np.uint8))
    return res_masks, labels_rw


def dummy_aug(mask):
    return [mask], None


augmentation_methods = [
    opencv_segmentation,
    split_mask_erode_dilate,
    skimage_watershed_segmentation,
    skimage_random_walker_segmentation
]


def masks_augmentation(masks):
    res_masks = []

    # always fill holes
    masks = [fill_holes_in_mask(m) for m in masks]  # NO DILATION

    method = augmentation_methods[np.random.randint(0, len(augmentation_methods))]
    print method.__name__

    for m in masks:
        try:
            segs, labels = method(m)
        except Exception as e:
            print(e)
            segs = [m]

        res_masks.extend(segs)

    k = np.random.choice([-2, -1, 1, 2])
    kernel = kernels[np.random.randint(0, len(kernels))]
    print k, kernel

    if k > 0:
        res_masks = [cv.dilate(m, kernel, iterations=k) for m in res_masks]
    elif k < 0:
        k = abs(k)
        res_masks = [cv.erode(m, kernel, iterations=k) for m in res_masks]

    return res_masks


def masks_augmentation_test(m, id=0):
    print('================', id)
    try:
        os.mkdir(os.path.join('.vis', str(id)))
    except:
        pass

    def apply(method, mask):
        print(method.__name__)
        try:
            M2, labels = method(m)
            print len(M2)
            method = method.__name__
            save(labels, os.path.join('.vis', str(id), method + '_labels.jpg'))
            save(m ^ np.sum(M2, 0), os.path.join('.vis', str(id), method + '_xor.jpg'))
            save(np.sum(M2, 0), os.path.join('.vis', str(id), method + '_res.jpg'))
        except Exception as e:
            print(e)

    apply(opencv_segmentation, m)
    apply(split_mask_erode_dilate, m)
    apply(skimage_watershed_segmentation, m)
    apply(skimage_random_walker_segmentation, m)


def test():
    import json, pickle
    D = json.load(open('data/stage1.json'))
    from pycocotools import mask as mask_util

    M = get_masks(D, 0)
    show(np.sum(M, 0))
    for i in range(100):
        if np.random.random() > 0.5:
            M2 = masks_augmentation(M)
            show(np.sum(M, 0) ^ np.sum(M2, 0))
        else:
            show(np.sum(M, 0))

    try:
        hole_masks = pickle.load(open('data/hole_masks.pkl'))
    except:
        hole_masks = []
        for a in D['annotations']:
            M = mask_util.decode(a['segmentation'])
            if check_holes(M):
                hole_masks.append((a, M))

    mask = hole_masks[-1][1]

    for idx, h in enumerate(hole_masks):
        mask = fill_holes_in_mask(h[1])
        masks_augmentation_test(mask, idx)

    for h in hole_masks:
        show(fill_holes_in_mask(h[1]))


def get_masks(D, image_id):
    from pycocotools import mask as mask_util

    M = []
    for a in D['annotations']:
        if a['image_id'] == image_id:
            M.append(mask_util.decode(a['segmentation']))

    return M


def check_holes(mask):
    print(mask.sum())
    mask_filled = fill_holes_in_mask(mask)
    mask_filled = mask_filled ^ mask
    print mask_filled.sum()
    if mask_filled.sum() > 5:
        return True


