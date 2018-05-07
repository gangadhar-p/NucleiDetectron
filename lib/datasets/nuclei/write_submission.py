import argparse
from pathlib import Path
import cPickle
from joblib import Parallel, delayed

from datasets.nuclei.mask_encoding import rle_encode


def load_results(version, version_id, iters):
    if version_id:
        RES_DIR = ROOT_DIR / 'results' / version / version_id / iters / 'test/nuclei_stage_1_test/generalized_rcnn/'
    else:
        # Example:
        # /detectron/lib/datasets/data/results/1_aug_gray_1_5_1_stage_2_v1/test/nuclei_stage_2_test/generalized_rcnn
        RES_DIR = ROOT_DIR / 'results' / version / 'test/nuclei_stage_1_test/generalized_rcnn'

    res_file = RES_DIR / 'detections.pkl'
    D = cPickle.load(open(res_file.as_posix()))
    return D, RES_DIR, ROOT_DIR


def write_submissions(version, version_id, iters, mask_area_threshold=30, accuracy_thresh=0.5, intersection_thresh=0.5):
    from pycocotools import mask as mask_util
    import numpy as np
    import cv2, os
    TOL = 0.00001

    D, RES_DIR, ROOT_DIR = load_results(version, version_id, iters)

    version_conf = '_'.join([str(i) for i in [mask_area_threshold, accuracy_thresh, intersection_thresh]])

    out_dir_res = RES_DIR / version_conf
    for d in [out_dir_res]:
        try:
            os.mkdir(d.as_posix())
        except Exception as e:
            print e

    D2 = {
        'all_boxes': [[], []],
        'all_segms': [[], []]
    }

    for boxs, segs in zip(D['all_boxes'][1], D['all_segms'][1]):
        segs = np.asarray(segs)
        sort_idx = (-boxs[:, 4]).argsort()  # descending order of confidance

        boxs = boxs[sort_idx]
        segs = segs[sort_idx]

        all_mask = None
        res_segs = []
        res_boxes = []
        all_mask_no_refine = None
        for b, rle in zip(boxs, segs):
            if b[-1] < accuracy_thresh:  # Skip if less than 80% accuracy
                continue

            mask_int_orig = mask_util.decode(rle)
            from scipy import ndimage
            mask_int = ndimage.morphology.binary_fill_holes(mask_int_orig.copy()).astype(np.uint8)

            mask = mask_int > 0
            mask_orig = mask_int_orig > 0

            if all_mask is None:
                all_mask = mask.copy()
                all_mask[:] = False
                all_mask_no_refine = mask_orig.copy()
                all_mask_no_refine[:] = False

            intersection = mask & all_mask

            area_inter = intersection.sum()
            if area_inter > 0:
                total_area = mask.sum()
                if float(area_inter) / (float(total_area) + TOL) > intersection_thresh:
                    continue

            mask = mask & ~all_mask
            if mask.sum() < mask_area_threshold:
                continue

            mask_int[~mask] = 0

            # add this to all_masks mask
            all_mask = mask | all_mask
            all_mask_no_refine = all_mask_no_refine | mask_orig
            res_segs.append(mask_util.encode(np.asarray(mask_int, order='F')))
            res_boxes.append(b)

        D2['all_boxes'][1].append(np.asarray(res_boxes))
        D2['all_segms'][1].append(res_segs)

    csv_res = ['ImageId,EncodedPixels']
    for idx, rles in enumerate(D2['all_segms'][1]):
        im_name = D['all_rois'][idx]['image'].rsplit('/', 1)[1].split('.')[0]
        for rle in rles:
            mask_int = mask_util.decode(rle)
            u_rle = rle_encode(mask_int)
            csv_res.append(','.join([im_name, ' '.join([str(x) for x in u_rle])]))

        if not rles:
            csv_res.append(','.join([im_name, '']))

    print('Wrote Submissions file!')
    version_id = version_id if version_id else ''
    iters = iters if iters else ''

    print(len(csv_res) - 1)
    submission_file = RES_DIR / (version + '_' + version_id + '_' + iters + '_thresh_' + version_conf + '.csv')
    with open(submission_file.as_posix(), 'w') as f:
        f.write('\n'.join(csv_res))

    result_tuples = []
    for roi, boxes, segms in zip(D['all_rois'], D2['all_boxes'][1], D2['all_segms'][1]):
        im_orig_path = roi['image']
        result_tuples.append((im_orig_path, boxes, segms))

    from utils.vis import save_im_fig
    Parallel(n_jobs=30)(delayed(save_im_fig)(t, out_dir_res.as_posix()) for t in result_tuples)


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '--results-root',
        dest='results_root',
        help='optional config file',
        default=None,
        type=str
    )
    parser.add_argument(
        '--run-version',
        dest='run_version',
        help='optional config file',
        default=None,
        type=str
    )
    parser.add_argument(
        '--version-id',
        dest='version_id',
        help='optional config file',
        default=None,
        type=str
    )
    parser.add_argument(
        '--iters',
        dest='iters',
        help='iteration',
        default=None,
        type=str
    )
    parser.add_argument(
        '--area-thresh',
        dest='area_threshold',
        help='iteration',
        default=None,
        type=int
    )
    parser.add_argument(
        '--acc-thresh',
        dest='acc_threshold',
        help='acc_threshold',
        default=None,
        type=float
    )
    parser.add_argument(
        '--intersection-thresh',
        dest='intersection_threshold',
        help='intersection_threshold',
        default=None,
        type=float
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)
    ROOT_DIR = Path(args.results_root)
    SUBMISSIONS_DIR = ROOT_DIR / 'submissions'

    write_submissions(version=args.run_version, version_id=args.version_id, iters=args.iters,
                      mask_area_threshold=args.area_threshold, accuracy_thresh=args.acc_threshold,
                      intersection_thresh=args.intersection_threshold)
