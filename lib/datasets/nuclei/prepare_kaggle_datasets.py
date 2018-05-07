from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
import sys

from pathlib import Path
import imageio
import json
import numpy as np

from datasets.nuclei.mask_encoding import encode_mask_to_rle, rle_to_string, rle_decode

from datasets.nuclei.nuclei_utils import image_ids_in

ROOT_DIR = Path('/media/gangadhar/DataSSD1TB/ROOT_DATA_DIR/')

'''
Folder Structure before running this script

ROOT_DATA_DIR/
    raw/
        kaggle-dsbowl-2018-dataset-fixes  # git clone https://github.com/lopuhin/kaggle-dsbowl-2018-dataset-fixes
        stage1_test  # unzip kaggle download stage1_train.zip
        stage1_solution.csv  # unzip kaggle download
        stage2_test_final  # unzip kaggle download stage2_test_final.zip
    
    Nuclei/
        annotations/
            stage_1_test  
            stage_1_train  
            stage_2_test
            classes.json
'''

RAW_INPUT_DIR = ROOT_DIR / 'raw'
RAW_TRAIN_DIR = RAW_INPUT_DIR / 'kaggle-dsbowl-2018-dataset-fixes' / 'stage1_train'
RAW_TEST_DIR = RAW_INPUT_DIR / 'stage1_test'
RAW_TEST2_DIR = RAW_INPUT_DIR / 'stage2_test_final'
RAW_STAGE_1_SOLUTION_FILE = RAW_INPUT_DIR / 'stage1_solution.csv'

DATASET_WORKING_DIR = ROOT_DIR / 'Nuclei'

STAGE_1_TRAIN_DIR = DATASET_WORKING_DIR / 'stage_1_train'
STAGE_1_TEST_DIR = DATASET_WORKING_DIR / 'stage_1_test'
STAGE_2_TEST_DIR = DATASET_WORKING_DIR / 'stage_2_test'

TRAIN_SPLIT_ANNOTATIONS_FILE = DATASET_WORKING_DIR / 'annotations/stage_1_local_train_split.json'
VAL_SPLIT_ANNOTATIONS_FILE = DATASET_WORKING_DIR / 'annotations/stage_1_local_val_split.json'
STAGE_1_TRAIN_ANNOTATIONS_FILE = DATASET_WORKING_DIR / 'annotations/stage1_train.json'
STAGE_1_TEST_ANNOTATIONS_FILE = DATASET_WORKING_DIR / 'annotations/stage_1_test.json'
STAGE_2_TEST_ANNOTATIONS_FILE = DATASET_WORKING_DIR / 'annotations/stage_2_test.json'

NUCLEI_CLASS_FILE_PATH = DATASET_WORKING_DIR / 'annotations/classes.json'


def load_stage1_solution():
    rle_masks = []
    with open(RAW_STAGE_1_SOLUTION_FILE.as_posix()) as f:
        for line in f:
            rle_masks.append(line.split(','))

    RLE = []
    for row in rle_masks:
        if row[0] == 'ImageId':
            continue
        row[1] = [int(x) for x in row[1].split(' ') if x]
        row[2] = int(row[2])
        row[3] = int(row[3])

        RLE.append(row)

    image_masks = {}

    for rle in RLE:
        m = rle_decode(rle_to_string(rle[1]), (rle[2], rle[3]), np.uint8)
        im_id = rle[0]

        if im_id not in image_masks:
            image_masks[im_id] = []

        image_masks[im_id].append(m)
    return image_masks


def load_images(root_dir, ids, get_masks=False):
    images = []
    masks = []
    image_sizes = []
    for id in ids:
        item_dir = root_dir / id
        image_path = item_dir / 'images' / (id + '.png')
        image = imageio.imread(str(image_path))

        # Enforce 3 channels for all images
        if len(image.shape) == 2:
            image = np.stack([image, image, image], -1)
        image = image[:, :, :3]  # remove the alpha channel as it is not used

        images.append(image)

        image_sizes.append(image.shape[:2])

        if get_masks:
            mask_sequence = []
            masks_dir = item_dir / 'masks'
            mask_paths = masks_dir.glob('*.png')
            for mask_path in mask_paths:
                mask = imageio.imread(str(mask_path))  # 0 and 255 values
                if len(mask.shape) > 2:
                    mask = np.sum(mask, axis=-1)
                mask = (mask > 0).astype(np.uint8)  # 0 and 1 values
                mask_sequence.append(mask)
            masks.append(mask_sequence)
    if get_masks:
        return images, masks, image_sizes
    else:
        return images, image_sizes


def get_image_data(masks, image_id, image_filename, size, class_name):
    im_metadata = {
        'file_name': image_filename + '.jpg',
        'height': size[0],
        'id': image_id,
        'width': size[1],
        'nuclei_class': class_name,
    }
    annotations = []
    global annotation_id
    for m in masks:
        annotations.append(encode_mask_to_rle(m, annotation_id, image_id))
        annotation_id += 1
    return im_metadata, annotations


def prepare_folder_structure():
    try:
        os.mkdir(STAGE_1_TEST_DIR.as_posix())
    except:
        pass
    try:
        os.mkdir(STAGE_2_TEST_DIR.as_posix())
    except:
        pass
    try:
        os.mkdir(STAGE_1_TRAIN_DIR.as_posix())
    except:
        pass
    try:
        os.mkdir((DATASET_WORKING_DIR / 'annotations').as_posix())
    except:
        pass


annotation_id = 0  # this is a global variable


def main():

    image_id_classes = json.load(open(NUCLEI_CLASS_FILE_PATH.as_posix()))
    train_image_ids = image_ids_in(RAW_TRAIN_DIR)
    test_image_ids = image_ids_in(RAW_TEST_DIR)
    test_stage_2_image_ids = image_ids_in(RAW_TEST2_DIR)

    train_images, train_masks, train_image_sizes = load_images(RAW_TRAIN_DIR, train_image_ids, True)
    test_images, test_image_sizes = load_images(RAW_TEST_DIR, test_image_ids, False)
    test_stage_2_images, test_stage_2_sizes = load_images(RAW_TEST2_DIR, test_stage_2_image_ids, False)

    stage_1_train_dataset = {
        'annotations': [],
        'categories': [
            {
                'id': 1,
                'name': 'Nucleus'
            }
        ],
        'images': [],
        'info': {
            'description': 'coco format Nuclei stage 1 train Dataset',
        }
    }

    stage_1_train_split_dataset = {
        'annotations': [],
        'categories': [
            {
                'id': 1,
                'name': 'Nucleus'
            }
        ],
        'images': [],
        'info': {
            'description': 'coco format Nuclei stage 1 train split Dataset',
        }
    }

    stage_1_val_split_dataset = {
        'annotations': [],
        'categories': [
            {
                'id': 1,
                'name': 'Nucleus'
            }
        ],
        'images': [],
        'info': {
            'description': 'coco format Nuclei stage 1 validation split Dataset',
        }
    }

    stage_1_test_dataset = {
        'annotations': [],
        'categories': [
            {
                'id': 1,
                'name': 'Nucleus'
            }
        ],
        'images': [],
        'info': {
            'description': 'coco format Nuclei stage 1 test Dataset',
        }
    }

    stage_2_test_dataset = {
        'annotations': [],
        'categories': [
            {
                'id': 1,
                'name': 'Nucleus'
            }
        ],
        'images': [],
        'info': {
            'description': 'coco format nucleous stage 1 test Dataset',
        }
    }

    prepare_folder_structure()

    im_count = 0

    for im, im_id, masks, sz in zip(train_images, train_image_ids, train_masks, train_image_sizes):
        class_name = image_id_classes[im_id]
        im_metadata, annotations_res = get_image_data(masks, im_count, im_id, sz, class_name)

        file_name = STAGE_1_TRAIN_DIR / (im_id + '.jpg')
        imageio.imsave(file_name.as_posix(), im)

        stage_1_train_dataset['images'].append(im_metadata)
        stage_1_train_dataset['annotations'].extend(annotations_res)

        if class_name in ['purple_purple_320_256_large', 'purple_purple_320_256_medium', 'purple_purple_320_256_small',
                          'purple_purple_320_256_sparce', 'purple_white_320_256', 'purple_white_320_256_long']:
            stage_1_val_split_dataset['images'].append(im_metadata)
            stage_1_val_split_dataset['annotations'].extend(annotations_res)
        else:
            stage_1_train_split_dataset['images'].append(im_metadata)
            stage_1_train_split_dataset['annotations'].extend(annotations_res)

        im_count += 1

    json.dump(stage_1_train_dataset, open(STAGE_1_TRAIN_ANNOTATIONS_FILE.as_posix(), 'w'))
    json.dump(stage_1_train_split_dataset, open(TRAIN_SPLIT_ANNOTATIONS_FILE.as_posix(), 'w'))
    json.dump(stage_1_val_split_dataset, open(VAL_SPLIT_ANNOTATIONS_FILE.as_posix(), 'w'))

    test_stage_1_masks = load_stage1_solution()
    for im, im_id, sz in zip(test_images, test_image_ids, test_image_sizes):
        class_name = image_id_classes[im_id]
        M = test_stage_1_masks[im_id]
        im_metadata, annotations_res = get_image_data(M, im_count, im_id, sz, class_name)

        file_name = STAGE_1_TEST_DIR / (im_id + '.jpg')
        imageio.imsave(file_name.as_posix(), im)

        stage_1_test_dataset['images'].append(im_metadata)
        stage_1_test_dataset['annotations'].extend(annotations_res)
        im_count += 1

    json.dump(stage_1_test_dataset, open(STAGE_1_TEST_ANNOTATIONS_FILE.as_posix(), 'w'))

    for im, im_id, sz in zip(test_stage_2_images, test_stage_2_image_ids, test_stage_2_sizes):
        class_name = 'Stage2'
        im_metadata, annotations_res = get_image_data([], im_count, im_id, sz, class_name)

        file_name = STAGE_2_TEST_DIR / (im_id + '.jpg')
        imageio.imsave(file_name.as_posix(), im)

        stage_2_test_dataset['images'].append(im_metadata)
        stage_2_test_dataset['annotations'].extend(annotations_res)
        im_count += 1

    json.dump(stage_2_test_dataset, open(STAGE_2_TEST_ANNOTATIONS_FILE.as_posix(), 'w'))


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare base datasets from kaggle')
    parser.add_argument(
        '--root-data-dir',
        dest='root_data_dir',
        help='Path to the root data dir',
        default=None,
        type=str
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    ROOT_DIR = Path(args.root_data_dir)
    main()
