import cPickle, os
from pycocotools import mask as mask_util
import numpy as np
import cv2
import utils.vis as vis_utils

from datasets.json_dataset import JsonDataset

TOL = 0.00000001


def load_dataset(dataset_name='nuclei_stage_1_local_val_split'):
    dataset = JsonDataset(dataset_name)
    roidb = dataset.get_roidb(gt=True)

    roidb_map = {}
    for roi in roidb:
        im_id = os.path.splitext(os.path.basename(roi['image']))[0]
        roidb_map[im_id] = roi
    return roidb_map


def load_predictions(roidb_map, detections_file):
    with open(detections_file) as f:
        D = cPickle.load(f)

    for roi in D['all_rois']:
        im_id = os.path.splitext(os.path.basename(roi['image']))[0]
        roi['segms'] = roidb_map[im_id]['segms']
        roi['image'] = roidb_map[im_id]['image']
    return D


def evaluate_all_nuclei_map(D):
    mAPs = []
    for gt_rois, pred_segs in zip(D['all_rois'], D['all_segms'][1]):
        if not pred_segs:
            mAPs.append(0)
            continue
        gt_masks = mask_util.decode(gt_rois['segms'])
        pred_masks = mask_util.decode(pred_segs)
        mAPs.append(evaluate_single_image(pred_masks, gt_masks))
    return mAPs


# source: https://www.kaggle.com/wcukierski/example-metric-implementation
def evaluate_single_image(predicted_masks, gt_masks):
    num_masks = gt_masks.shape[-1]
    height, width, _ = gt_masks.shape

    # Make a ground truth label image (pixel value is index of object label)
    labels = np.zeros((height, width), np.uint16)
    for index in range(0, num_masks):
        labels[gt_masks[:, :, index] > 0] = index + 1

    y_pred = np.zeros((height, width), np.uint16)
    for index in range(0, predicted_masks.shape[-1]):
        y_pred[predicted_masks[:, :, index] > 0] = index + 1

    true_objects = len(np.unique(labels))
    pred_objects = len(np.unique(y_pred))

    # Compute intersection between all objects
    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins=true_objects)[0]
    area_pred = np.histogram(y_pred, bins=pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:, 1:]
    union = union[1:, 1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1  # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        p = float(tp) / float(tp + fp + fn)
        prec.append(p)
    mAP = np.mean(prec)
    return mAP


def decode_masks(D):
    D2 = {
        'all_boxes': [[], []],
        'all_segms': [[], []],
        'all_masks': [[], []],
        'all_rois': D['all_rois']
    }
    for boxs, segs in zip(D['all_boxes'][1], D['all_segms'][1]):
        if not segs:
            D2['all_boxes'][1].append(boxs)
            D2['all_segms'][1].append(segs)
            D2['all_masks'][1].append([])
            continue

        M = mask_util.decode(segs)
        segs = np.asarray(segs)

        sort_idx = (-boxs[:, 4]).argsort()  # descending order of confidance
        boxs = boxs[sort_idx]
        segs = segs[sort_idx]
        M = M[:, :, sort_idx]

        M = M > 0

        D2['all_boxes'][1].append(boxs)
        D2['all_segms'][1].append(segs)
        D2['all_masks'][1].append(M)

    return D2


def decode_masks_no_sort(D):
    D['all_masks'] = [[], []]
    for segs in D['all_segms'][1]:
        print 'Seg Len:', len(segs)
        M = mask_util.decode(segs)
        M = M > 0
        D['all_masks'][1].append(M)


def filter_detections(D, accuracy_thresh, mask_area_threshold, intersection_thresh):
    D2 = {
        'all_boxes': [[], []],
        'all_segms': [[], []],
        'all_rois': D['all_rois'],
    }

    for boxs, segs, M in zip(D['all_boxes'][1], D['all_segms'][1], D['all_masks'][1]):
        all_mask = None
        res_segs = []
        res_boxes = []

        for idx in range(len(boxs)):
            b, rle, m = boxs[idx], segs[idx], M[:, :, idx]

            if b[-1] < accuracy_thresh:  # Skip if less than 80% accuracy
                break

            if all_mask is None:
                all_mask = m.copy()
                all_mask[:] = False

            intersection = m & all_mask
            area_inter = intersection.sum()
            if area_inter > 0:
                total_area = m.sum()
                if float(area_inter) / (float(total_area) + TOL) > intersection_thresh:
                    continue

                mask = m & ~all_mask
                total_area = mask.sum()
                if total_area < mask_area_threshold:
                    continue

                res_segs.append(mask_util.encode(np.asarray(mask, dtype=np.uint8, order='F')))
            else:
                total_area = m.sum()
                if total_area < mask_area_threshold:
                    continue
                res_segs.append(rle)

            res_boxes.append(b)

            # add this to all_masks mask
            all_mask = m | all_mask

        D2['all_boxes'][1].append(np.asarray(res_boxes))
        D2['all_segms'][1].append(res_segs)

    return D2


def run_evaluation(D, a, m, inter, iters, out_dir):
    D2 = filter_detections(D, a, m, inter)
    mAp = evaluate_all_nuclei_map(D2)

    im_results = [
        {
            'Precision': ap,
            'image_id': gt_rois['image'],
            'nuclei_class': gt_rois['nuclei_class'],
            'NumberOfDetections': len(dets)
        }
        for ap, gt_rois, dets in
        zip(mAp, D2['all_rois'], D2['all_boxes'][1])
    ]
    result = {'accuracy_threshold': a,
              'mask_area_threshold': m,
              'intersection_thresh': inter,
              'mAp': np.mean(mAp),
              'im_results': im_results}

    params = ', '.join([str(i) for i in [iters, a, m, inter]])
    print params, np.mean(mAp)

    for ap, gt_roi, boxes, segms in zip(mAp, D2['all_rois'], D2['all_boxes'][1], D2['all_segms'][1]):
        visualize_im_masks(gt_roi, boxes, segms, out_dir, True, ap)

    return result


def visualize_im_masks(entry, boxes, segms, output_dir, show_class=True, ap=None):
    im_name = os.path.splitext(os.path.basename(entry['image']))[0]
    im = cv2.imread(entry['image'])

    if ap:
        im_name = "{:.5f}_{}".format(ap, im_name)

    classes = [1] * len(boxes)

    vis_utils.vis_one_image(
        im[:, :, ::-1],
        im_name,
        os.path.join(output_dir, 'vis_sorted'),
        boxes=boxes,
        segms=segms,
        keypoints=None,
        thresh=0.0,
        box_alpha=0.8,
        dataset=entry['dataset'],
        show_class=show_class,
        ext='pdf',
        classes=classes
    )


def visualize_ground_truth(dataset_name='nuclei_stage_1_local_val_split', output_dir='vis'):
    TOL = 0.00000001
    dataset = JsonDataset(dataset_name)
    roidb = dataset.get_roidb(gt=True)
    for entry in roidb:
        boxes = entry['boxes']
        boxes = np.append(boxes, np.ones((len(boxes), 2)), 1)
        segms = entry['segms']
        visualize_im_masks(entry, boxes, segms, output_dir, show_class=False)
