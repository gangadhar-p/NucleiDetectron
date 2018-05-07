import argparse

from pathlib import Path
from datasets.nuclei.evaluation import load_dataset, load_predictions, run_evaluation, decode_masks

ROOT_DIR = Path('/media/gangadhar/DataSSD1TB/ROOT_DATA_DIR/')


def evaluate_results(run_version, iters):
    results_dir = ROOT_DIR / 'results/{}'.format(run_version)

    detections_file = results_dir / '{}/test/nuclei_stage_1_local_val_split/generalized_rcnn/detections.pkl'.format(iters)
    results_file = results_dir / '{}/test/nuclei_stage_1_local_val_split/generalized_rcnn/results_grid_search.json'.format(iters)

    D = load_predictions(roidb_map, detections_file.as_posix())
    D = decode_masks(D)

    visualization_folder = results_dir / '{}/test/nuclei_stage_1_local_val_split/generalized_rcnn/'.format(iters)

    results = postprocess(D, iters, visualization_folder.as_posix())

    with open(results_file.as_posix(), 'w') as f:
        import json
        json.dump(results, f)

    return results


def postprocess(D, iters, output_dir):
    accuracy_thresholds = [0.9]

    mask_area_threshold = [15]
    intersection_thresh = [0.3]

    results = []
    for a in accuracy_thresholds:
        for m in mask_area_threshold:
            for inter in intersection_thresh:
                results.append(run_evaluation(D, a, m, inter, iters, output_dir))

    return results


def parse_args():
    parser = argparse.ArgumentParser(description='Postprocess')
    parser.add_argument(
        '--run-version',
        dest='run_version',
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
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)

    roidb_map = load_dataset()
    evaluate_results(args.run_version, args.iters)
