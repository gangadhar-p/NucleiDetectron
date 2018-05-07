from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import cv2
import imageio
import numpy as np
import cv2 as cv
import scipy
import xmltodict
from pycocotools import mask as mask_util


# ref: https://www.kaggle.com/stainsby/fast-tested-rle-and-input-routines
def rle_encode(mask):
    pixels = mask.T.flatten()
    # We need to allow for cases where there is a '1' at either end of the sequence.
    # We do this by padding with a zero at each end when needed.
    use_padding = False
    if pixels[0] or pixels[-1]:
        use_padding = True
        pixel_padded = np.zeros([len(pixels) + 2], dtype=pixels.dtype)
        pixel_padded[1:-1] = pixels
        pixels = pixel_padded
    rle = np.where(pixels[1:] != pixels[:-1])[0] + 2
    if use_padding:
        rle = rle - 1
    rle[1::2] = rle[1::2] - rle[:-1:2]
    return rle


def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)


# This is copied from https://www.kaggle.com/paulorzp/run-length-encode-and-decode.
# Thanks to Paulo Pinto.
def rle_decode(rle_str, mask_shape, mask_dtype):
    s = rle_str.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    mask = np.zeros(np.prod(mask_shape), dtype=mask_dtype)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask.reshape(mask_shape[::-1]).T


def encode_mask_to_poly(mask, mask_id, image_id):
    if len(mask.shape) == 3:
        mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)

    kernel = np.ones((2, 2), np.uint8)

    mask = cv.dilate(mask, kernel, iterations=1)
    _, C, h = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    seg = [[float(x) for x in contour.flatten()] for contour in C]
    seg = [cont for cont in seg if len(cont) > 4]  # filter all polygons that are boxes
    rle = mask_util.frPyObjects(seg, mask.shape[0], mask.shape[1])

    return {
        'area': float(sum(mask_util.area(rle))),
        'bbox': list(mask_util.toBbox(rle)[0]),
        'category_id': 1,
        'id': mask_id,
        'image_id': image_id,
        'iscrowd': 0,
        'segmentation': seg
    }


def encode_mask_to_rle(mask, mask_id, image_id):
    seg = mask_util.encode(np.asarray(mask, order='F'))
    return encode_rle(seg, mask_id, image_id)


def encode_rle(rle, mask_id, image_id):
    rle['counts'] = rle['counts'].decode('utf-8')
    return {
        'image_id': image_id,
        'segmentation': rle,
        'category_id': 1,
        'id': mask_id,
        'area': int(mask_util.area(rle)),
        'bbox': list(mask_util.toBbox(rle)),
        'iscrowd': 0
    }


def regions_to_rle(regions, shape):
    R = [r.flatten() for r in regions]
    rle = mask_util.frPyObjects(R, shape[0], shape[1])
    return rle


def parse_xml_annotations(file_path):
    with open(file_path) as f:
        xml = f.read()

    ann = xmltodict.parse(xml)
    regions = []
    if isinstance(ann['Annotations']['Annotation'], list):
        print('Found Multiple regions')
        for a in ann['Annotations']['Annotation']:
            if 'Regions' in a and 'Region' in a['Regions']:
                for region in a['Regions']['Region']:
                    vertices = []
                    for v in region['Vertices']['Vertex']:
                        vertices.append([float(v['@X']), float(v['@Y'])])

                    regions.append(np.asarray(vertices))
    else:
        for region in ann['Annotations']['Annotation']['Regions']['Region']:
            vertices = []
            for v in region['Vertices']['Vertex']:
                vertices.append([float(v['@X']), float(v['@Y'])])

            regions.append(np.asarray(vertices))

    return regions


def filter_contours(contours, H):
    C = []
    i = 0
    while i != -1:
        j = H[i][2]
        while j != -1:
            C.append(contours[j])
            j = H[j][0]
        i = H[i][0]


kernel = np.ones((3, 3), np.uint8)


def dedupe_contours(rles, dataset):
    M = mask_util.decode(rles)
    all_mask = M[:, :, 0].copy()
    all_mask[:] = False

    areas = np.sum(M, (0, 1))
    sort_idx = areas.argsort()
    areas = areas[sort_idx]
    M = M[:, :, sort_idx]

    res = []
    im_size = M.shape[0] * M.shape[1]
    for idx in range(M.shape[-1]):
        if areas[idx] < 30 or areas[idx] > im_size * 0.5:
            continue

        m = M[:, :, idx]
        intersection = m & all_mask
        area_inter = intersection.sum()
        if area_inter > 30:
            continue
        else:
            mask = m & ~all_mask
            total_area = mask.sum()
            if total_area < 30:
                continue

        if dataset not in ['2009_ISBI_2DNuclei', 'cluster_nuclei']:
            m = cv.dilate(m, kernel, iterations=1)

        all_mask = m | all_mask
        res.append(m)

    if not res:
        return None

    M2 = np.stack(res).transpose((1, 2, 0))

    if dataset == '2009_ISBI_2DNuclei':
        M2 = scipy.ndimage.zoom(M2, (0.4, 0.4, 1), order=1)

    rles = mask_util.encode(np.asarray(M2, dtype=np.uint8, order='F'))
    return rles


def parse_segments_from_outlines(outline_path, dataset):
    if dataset == 'BBBC006':
        import imread
        masks = imread.imread(outline_path)
        rles = []
        for idx in range(1, masks.max() + 1):
            rles.append(mask_util.encode(np.asarray(masks == idx, dtype=np.uint8, order='F')))
        return rles

    if dataset == 'BBBC020':
        out_dir, prefix = outline_path.rsplit('/', 1)
        files = os.listdir(out_dir)
        masks = []
        for f_name in files:
            if prefix in f_name:
                m = imageio.imread(os.path.join(out_dir, f_name))
                m[m > 0] = 1

                m = scipy.ndimage.zoom(m, (0.4, 0.4), order=1)
                masks.append(m)
        rles = []
        for m in masks:
            rles.append(mask_util.encode(np.asarray(m, dtype=np.uint8, order='F')))
        return rles

    if dataset == '2009_ISBI_2DNuclei':
        import imread
        outlines = imread.imread(outline_path)
    else:
        outlines = imageio.imread(outline_path)

    if dataset == 'cluster_nuclei' or dataset == '2009_ISBI_2DNuclei':
        outlines[outlines != [255, 0, 0]] = 0
        imgray = cv2.cvtColor(outlines, cv2.COLOR_RGB2GRAY)
        ret, thresh = cv2.threshold(imgray, 0, 255, 0)
    elif dataset == 'BBBC007':
        thresh = np.asarray(outlines, np.uint8)
    elif dataset == 'BBBC018':
        thresh = outlines

    thresh[0, :] = 1
    thresh[:, 0] = 1
    thresh[:, -1] = 1
    thresh[-1, :] = 1

    im, contours, hierarchy = cv2.findContours(thresh,
                                               cv2.RETR_CCOMP,
                                               cv2.CHAIN_APPROX_SIMPLE)

    seg = [[float(x) for x in c.flatten()] for c in contours]
    seg = [cont for cont in seg if len(cont) > 4]  # filter all polygons that are boxes
    if not seg:
        return []

    rles = mask_util.frPyObjects(seg, outlines.shape[0], outlines.shape[1])
    rles = dedupe_contours(rles, dataset)
    return rles
