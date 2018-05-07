from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
import sys

from pathlib import Path
from pycocotools import mask as mask_util
import imageio
import json
import numpy as np
import scipy.ndimage

from datasets.nuclei.mask_encoding import encode_rle, parse_segments_from_outlines, parse_xml_annotations, \
    regions_to_rle
from datasets.nuclei.nuclei_utils import rescale_0_255, image_ids_in

ROOT_DIR = Path('/media/gangadhar/DataSSD1TB/ROOT_DATA_DIR/')
DATASET_WORKING_DIR = ROOT_DIR / 'Nuclei'


def load_images(raw_train_images_dir, raw_train_annotations_dir, ids, dataset):
    images = []
    rles_list = []
    image_sizes = []
    im_ids = []
    for id in ids:
        im_path = str(raw_train_images_dir / id)
        if dataset == 'BBBC018':
            from bioformats import ImageReader
            with ImageReader(im_path) as reader:
                image = reader.read()
                image = np.asarray(rescale_0_255(image), np.uint8)
        else:
            image = imageio.imread(im_path)

        if len(image.shape) == 2:
            image = np.stack([image, image, image]).transpose((1, 2, 0))

        image = image[:, :, :3]  # remove the alpha channel as it is not used

        if dataset == 'BBBC007':
            outline_path = (raw_train_annotations_dir / id).as_posix()
        elif dataset == 'cluster_nuclei':
            outline_path = (raw_train_annotations_dir / (id.split('.')[0] + '(label).bmp')).as_posix()
        elif dataset == 'BBBC018':
            outline_path = (raw_train_annotations_dir / (id.split('-')[0] + '-nuclei.png')).as_posix()
        elif dataset == 'BBBC020':
            outline_path = (raw_train_annotations_dir / (id.split('_')[0] + '_c5')).as_posix()
            image = scipy.ndimage.zoom(image, (0.4, 0.4, 1), order=3)
        elif dataset == '2009_ISBI_2DNuclei':
            outline_path = (raw_train_annotations_dir / (id.split('.')[0] + '.xcf')).as_posix()
            image = scipy.ndimage.zoom(image, (0.4, 0.4, 1), order=3)

        if dataset in ['BBBC018', '2009_ISBI_2DNuclei'] and not os.path.isfile(outline_path):
            continue
        if dataset == 'BBBC006':
            outline_path = (raw_train_annotations_dir / (id.split(':', 1)[1].rsplit('_', 1)[0] + '.png')).as_posix()

        rles = parse_segments_from_outlines(outline_path, dataset)
        if not rles:
            continue

        rles_list.append(rles)
        images.append(image)
        image_sizes.append(image.shape[:2])
        im_ids.append(id)

    return images, rles_list, image_sizes, im_ids


def load_images_benchmark_tissue(root_dir, ids, output_train_images_dir):
    images = []
    rles_list = []
    image_sizes = []
    for id in ids:
        image = imageio.imread(str(root_dir / id))
        image = image[:, :, :3]  # remove the alpha channel as it is not used
        images.append(image)
        image_sizes.append(image.shape[:2])

        mask_file_path = (output_train_images_dir / (id.split('.')[0] + '.xml')).as_posix()
        regions = parse_xml_annotations(mask_file_path)
        rles = regions_to_rle(regions, image.shape[0:2])
        rles_list.append(rles)

    return images, rles_list, image_sizes


def tile_image(I, sz=512, resize=None, order=3):
    height, width, _ = I.shape
    import scipy.ndimage

    chunks = []
    names = []
    for h in range(0, height, sz):
        for w in range(0, width, sz):
            w_end = w + sz
            h_end = h + sz
            c = I[w:w_end, h:h_end]
            n = '{}_{}_x_{}_{}'.format(w, w_end, h, h_end)
            if resize:
                c = scipy.ndimage.zoom(c, (resize / float(sz), resize / float(sz), 1), order=order)
            chunks.append(c)
            names.append(n)

    return chunks, names


def get_all_tiles(I, sizes, resize, order=3):
    tiles = []
    names = []

    for sz in sizes:  # [128, 256, 512, 1000]:
        c, n = tile_image(I, sz, resize, order=order)
        tiles.extend(c)
        names.extend(n)
        print('chunk created')
    return tiles, names


def filter_masks(M):
    masks = []
    for idx in range(M.shape[2]):  # for each mask channel
        if M[:, :, idx].sum() < 5:
            continue
        masks.append(M[:, :, idx])
    if masks:
        return True, np.stack(masks).transpose((1, 2, 0))  # put channels back to place
    return False, None


def convert_union_mask_to_masks(mask_union):
    from skimage import measure
    assert mask_union.shape[2] == 1

    blobs_labels = measure.label(mask_union[:, :, 0], background=0)
    masks = []
    for idx in range(1, blobs_labels.max() + 1):  # for each mask channel
        masks.append(blobs_labels == idx)

    return np.stack(masks).transpose((1, 2, 0))  # put channels back to place


def load_image(raw_train_dir, raw_train_annotations_dir, im_id, dataset_name, class_name_base, im_count, train_dir):
    image = imageio.imread(str(raw_train_dir / im_id))
    image = image[:, :, :3]  # remove the alpha channel as it is not used

    if dataset_name == 'nuclei_partial_annotations':
        mask_file_path = (raw_train_annotations_dir / (im_id.rsplit('_', 1)[0] + '_mask.png')).as_posix()
    elif dataset_name == 'TNBC_NucleiSegmentation':
        mask_file_path = (raw_train_annotations_dir / (im_id)).as_posix()
    else:
        mask_file_path = None

    if not os.path.isfile(mask_file_path):
        return None, None, im_count

    if dataset_name in ['nuclei_partial_annotations', 'TNBC_NucleiSegmentation']:
        mask_union = imageio.imread(mask_file_path)
        mask_union[mask_union > 0] = 1
        M = np.expand_dims(mask_union, -1)
    else:
        M = None  # not supported yet

    if dataset_name in ['nuclei_partial_annotations']:
        mtiles, _ = get_all_tiles(M, [512], 512, order=3)
        tiles, names = get_all_tiles(image, [512], 512, order=3)
    elif dataset_name in ['TNBC_NucleiSegmentation']:
        mtiles, _ = get_all_tiles(M, [512], 512, order=3)
        tiles, names = get_all_tiles(image, [512], 512, order=3)
    else:
        mtiles, tiles, names = None, None, None

    im_metadata_list, annotations_list = [], []

    for t, m, n in zip(tiles, mtiles, names):
        success, t_m = filter_masks(m)
        if success:
            t_id = '{}.{}'.format(im_id, n)
            t_cls = '{}_{}'.format(class_name_base, np.random.choice([0, 1]))
            t_sz = t.shape[:2]

            if dataset_name in ['nuclei_partial_annotations', 'TNBC_NucleiSegmentation']:
                t_m = convert_union_mask_to_masks(t_m)

            t_rle = mask_util.encode(np.asarray(t_m, dtype=np.uint8, order='F'))
            if len(t_rle) < 3:
                continue

            im_metadata, annotations_res = get_image_data(t, t_rle, im_count, t_id, t_sz, t_cls, train_dir)
            im_metadata_list.append(im_metadata)
            annotations_list.extend(annotations_res)
            im_count += 1

    return im_metadata_list, annotations_list, im_count


def preprocess_as_tiles(orig_images, orig_rles_list, orig_im_ids):
    images = []
    rles_list = []
    image_sizes = []
    image_names = []

    for I, rles, im_name in zip(orig_images, orig_rles_list, orig_im_ids):
        M = mask_util.decode(rles)

        mtiles, _ = get_all_tiles(M, [512], 512, order=1)
        tiles, names = get_all_tiles(I, [512], 512, order=3)

        for t, m, n in zip(tiles, mtiles, names):
            success, m = filter_masks(m)
            if success:
                rles_list.append(mask_util.encode(np.asarray(m, order='F')))
                images.append(t)
                image_sizes.append(t.shape[:2])
                image_names.append('{}:{}'.format(im_name, n))
                print('Image')
            else:
                print('Failed Image')
    return images, rles_list, image_sizes, image_names


def get_image_data(image, rles, image_id, image_filename, size, class_name, train_image_dir):
    im_metadata = {
        'file_name': image_filename + '.jpg',
        'height': size[0],
        'id': image_id,
        'width': size[1],
        'nuclei_class': class_name,
        # 'is_grey_scale': is_grey_scale_mat(image)
    }

    annotations = []
    global annotation_id
    for rle in rles:
        encoded_segment = encode_rle(rle, annotation_id, image_id)
        if encoded_segment['area'] > 0:
            annotations.append(encoded_segment)
            annotation_id += 1
        else:
            from pprint import pprint as pp
            pp(encoded_segment)

    if annotations:
        file_name = train_image_dir / (image_filename + '.jpg')
        imageio.imsave(file_name.as_posix(), image)

    return im_metadata, annotations


annotation_id = 0


def prepare_cluster_nuclei():
    dataset_name = 'cluster_nuclei'

    # ref: https://imagej.nih.gov/ij/plugins/ihc-toolbox/index.html
    # download and extract https://www.dropbox.com/s/9knzkp9g9xt6ipb/cluster%20nuclei.zip?dl=0

    raw_input_dir = ROOT_DIR / 'raw_external/cluster nuclei'

    raw_train_images_dir = raw_input_dir / 'original'
    raw_train_annotations_dir = raw_input_dir / 'label'

    train_images_output_dir = DATASET_WORKING_DIR / dataset_name

    try:
        os.mkdir(train_images_output_dir.as_posix())
    except:
        pass

    annotations_file_path = DATASET_WORKING_DIR / 'annotations/cluster_nuclei.json'

    im_ids = image_ids_in(raw_train_images_dir)
    images, rle_lists, image_sizes, train_image_ids = load_images(raw_train_images_dir,
                                                                  raw_train_annotations_dir,
                                                                  im_ids,
                                                                  dataset_name)

    im_count = 0
    dataset_structure = {
        'annotations': [],
        'categories': [
            {
                'id': 1,
                'name': 'Nucleus'
            }
        ],
        'images': [],
        'info': {
            'description': 'coco format Nuclei external Clustered_Nuclei Dataset',
        }
    }

    for im, im_id, masks, sz in zip(images, train_image_ids, rle_lists, image_sizes):
        im_metadata, annotations_res = get_image_data(im, masks, im_count, im_id, sz, 'Clustered_Nuclei',
                                                      train_images_output_dir)
        dataset_structure['images'].append(im_metadata)
        dataset_structure['annotations'].extend(annotations_res)
        im_count += 1

    json.dump(dataset_structure, open(annotations_file_path.as_posix(), 'w'))


def prepare_BBBC007():

    dataset_name = 'BBBC007'
    raw_input_dir = ROOT_DIR / 'raw_external/'

    # ref: https://data.broadinstitute.org/bbbc/image_sets.html
    # download this from the above link and extract
    raw_train_images_dir = raw_input_dir / 'BBBC007_v1_images'
    raw_train_annotations_dir = raw_input_dir / 'BBBC007_v1_outlines'

    train_images_output_dir = DATASET_WORKING_DIR / dataset_name

    try:
        os.mkdir(train_images_output_dir.as_posix())
    except:
        pass

    annotations_file = DATASET_WORKING_DIR / 'annotations/BBBC007.json'

    image_ids = image_ids_in(raw_train_images_dir)

    images, rle_lists, image_sizes, train_image_ids = load_images(raw_train_images_dir,
                                                                  raw_train_annotations_dir,
                                                                  image_ids,
                                                                  dataset=dataset_name)

    class_name = 'white_black_large_BBBC007'

    im_count = 0
    dataset_structure = {
        'annotations': [],
        'categories': [
            {
                'id': 1,
                'name': 'Nucleus'
            }
        ],
        'images': [],
        'info': {
            'description': 'coco format nucleous external BBBC007 Dataset',
        }
    }

    for im, im_id, masks, sz in zip(images, train_image_ids, rle_lists, image_sizes):
        im_metadata, annotations_res = get_image_data(im, masks, im_count,
                                                      im_id, sz, class_name, train_images_output_dir)
        dataset_structure['images'].append(im_metadata)
        dataset_structure['annotations'].extend(annotations_res)

        im_count += 1

    json.dump(dataset_structure, open(annotations_file.as_posix(), 'w'))


def prepare_BBBC018():
    dataset_name = 'BBBC018'
    raw_input_dir = ROOT_DIR / 'raw_external/'

    # ref: https://data.broadinstitute.org/bbbc/image_sets.html
    # download this from the above link and extract
    raw_train_images_dir = raw_input_dir / 'BBBC018_v1_images'
    raw_train_annotations_dir = raw_input_dir / 'BBBC018_v1_outlines'

    train_images_output_dir = DATASET_WORKING_DIR / dataset_name

    try:
        os.mkdir(train_images_output_dir.as_posix())
    except:
        pass

    annotations_file_path = DATASET_WORKING_DIR / 'annotations/BBBC018.json'

    im_ids = image_ids_in(raw_train_images_dir)

    im_ids = [id for id in im_ids if 'DNA' in id]

    import javabridge
    import bioformats
    javabridge.start_vm(class_path=bioformats.JARS)

    images, rle_lists, image_sizes, train_im_ids = load_images(raw_train_images_dir,
                                                               raw_train_annotations_dir,
                                                               im_ids,
                                                               dataset=dataset_name)

    class_name = 'white_black_small_BBBC018'

    im_count = 0
    dataset_structure = {
        'annotations': [],
        'categories': [
            {
                'id': 1,
                'name': 'Nucleus'
            }
        ],
        'images': [],
        'info': {
            'description': 'coco format Nuclei external BBBC018 Dataset',
        }
    }

    for im, im_id, masks, sz in zip(images, train_im_ids, rle_lists, image_sizes):
        im_metadata, annotations_res = get_image_data(im, masks, im_count,
                                                      im_id, sz, class_name, train_images_output_dir)
        dataset_structure['images'].append(im_metadata)
        dataset_structure['annotations'].extend(annotations_res)

        im_count += 1

    json.dump(dataset_structure, open(annotations_file_path.as_posix(), 'w'))


def prepare_BBBC020():
    dataset_name = 'BBBC020'
    raw_input_dir = ROOT_DIR / 'raw_external/'

    # ref: https://data.broadinstitute.org/bbbc/image_sets.html
    # download this from the above link and extract
    raw_train_images_dir = raw_input_dir / 'BBBC020_v1_images'
    raw_train_annotations_dir = raw_input_dir / 'BBBC020_v1_outlines_nuclei'

    train_images_output_dir = DATASET_WORKING_DIR / dataset_name
    try:
        os.mkdir(train_images_output_dir.as_posix())
    except:
        pass
    
    annotations_file_path = DATASET_WORKING_DIR / 'annotations/BBBC020.json'

    train_im_ids = image_ids_in(raw_train_images_dir)

    im_ids = [id for id in train_im_ids if '_c1' not in id]
    im_ids_c5 = [id for id in im_ids if 'c1' not in id]
    im_ids_c1_c5 = [id for id in im_ids if 'c1' in id]

    images, rle_lists, image_sizes, train_im_ids_c5 = load_images(raw_train_images_dir,
                                                                  raw_train_annotations_dir,
                                                                  im_ids_c5,
                                                                  dataset=dataset_name)

    images2, rle_list2, image_sizes2, train_im_ids_c1_c5 = load_images(raw_train_images_dir,
                                                                       raw_train_annotations_dir,
                                                                       im_ids_c1_c5,
                                                                       dataset=dataset_name)

    class_name = 'black_color_BBBC020'

    im_count = 0
    dataset_structure = {
        'annotations': [],
        'categories': [
            {
                'id': 1,
                'name': 'Nucleus'
            }
        ],
        'images': [],
        'info': {
            'description': 'coco format Nuclei external BBBC020 Dataset',
        }
    }

    for im, im_id, masks, sz in zip(images, train_im_ids_c5, rle_lists, image_sizes):
        im_metadata, annotations_res = get_image_data(im, masks, im_count,
                                                      im_id, sz, class_name, train_images_output_dir)
        dataset_structure['images'].append(im_metadata)
        dataset_structure['annotations'].extend(annotations_res)

        im_count += 1

    for im, im_id, masks, sz in zip(images2, train_im_ids_c1_c5, rle_list2, image_sizes2):
        im_metadata, annotations_res = get_image_data(im, masks, im_count,
                                                      im_id, sz, class_name, train_images_output_dir)
        dataset_structure['images'].append(im_metadata)
        dataset_structure['annotations'].extend(annotations_res)

        im_count += 1

    json.dump(dataset_structure, open(annotations_file_path.as_posix(), 'w'))


def prepare_2009_ISBI_2DNuclei_code_data():
    dataset_name = '2009_ISBI_2DNuclei'
    raw_input_dir = ROOT_DIR / 'raw_external/'

    # ref: http://murphylab.web.cmu.edu/data/2009_ISBI_Nuclei.html
    # download and extract http://murphylab.web.cmu.edu/data/2009_ISBI_2DNuclei_code_data.tgz
    raw_train_images_dir = raw_input_dir / '2009_ISBI_2DNuclei_code_data/data/images/dna-images'
    raw_train_annotations_dir = raw_input_dir / '2009_ISBI_2DNuclei_code_data/data/images/segmented-lpc'

    train_images_output_dir = DATASET_WORKING_DIR / dataset_name

    try:
        os.mkdir(train_images_output_dir.as_posix())
    except:
        pass

    annotations_file_path = DATASET_WORKING_DIR / 'annotations/2009_ISBI_2DNuclei.json'

    im_ids = image_ids_in(raw_train_images_dir)

    images, rle_lists, image_sizes, train_im_ids = load_images(raw_train_images_dir,
                                                               raw_train_annotations_dir,
                                                               im_ids,
                                                               dataset=dataset_name)

    class_name = 'white_black_large_2009_ISBI_2DNuclei'

    im_count = 0
    dataset_structure = {
        'annotations': [],
        'categories': [
            {
                'id': 1,
                'name': 'Nucleus'
            }
        ],
        'images': [],
        'info': {
            'description': 'coco format Nuclei external 2009_ISBI_2DNuclei Dataset',
        }
    }

    for im, im_id, masks, sz in zip(images, train_im_ids, rle_lists, image_sizes):
        im_metadata, annotations_res = get_image_data(im, masks, im_count,
                                                      im_id, sz, class_name, train_images_output_dir)
        dataset_structure['images'].append(im_metadata)
        dataset_structure['annotations'].extend(annotations_res)

        im_count += 1

    json.dump(dataset_structure, open(annotations_file_path.as_posix(), 'w'))


def prepare_nuclei_partial_annotations():
    dataset_name = 'nuclei_partial_annotations'

    # ref http://www.andrewjanowczyk.com/use-case-1-nuclei-segmentation/
    # download and extract http://andrewjanowczyk.com/wp-static/nuclei.tgz
    raw_input_dir = ROOT_DIR / 'raw_external/nuclei_partial_annotations'

    raw_train_images_dir = raw_input_dir / 'images'
    raw_train_annotations_dir = raw_input_dir / 'masks'

    train_images_output_dir = DATASET_WORKING_DIR / dataset_name
    try:
        os.mkdir(train_images_output_dir.as_posix())
    except:
        pass
    annotations_file_path = DATASET_WORKING_DIR / 'annotations/nuclei_partial_annotations.json'

    try:
        os.mkdir(train_images_output_dir.as_posix())
    except:
        pass

    im_ids = image_ids_in(raw_train_images_dir)

    class_name_base = 'nuclei_partial_annotations'
    im_count = 0

    dataset_structure = {
        'annotations': [],
        'categories': [
            {
                'id': 1,
                'name': 'Nucleus'
            }
        ],
        'images': [],
        'info': {
            'description': 'coco format Nuclei external nuclei_partial_annotations Dataset',
        }
    }

    for im_id in im_ids:
        im_metadata, annotations_res, im_count = load_image(raw_train_images_dir, raw_train_annotations_dir,
                                                            im_id, dataset_name, class_name_base,
                                                            im_count, train_images_output_dir)
        if not im_metadata:
            continue

        dataset_structure['images'].extend(im_metadata)
        dataset_structure['annotations'].extend(annotations_res)

        im_count += 1

    json.dump(dataset_structure, open(annotations_file_path.as_posix(), 'w'))


def prepare_TNBC_NucleiSegmentation():
    dataset_name = 'TNBC_NucleiSegmentation'

    # ref: https://zenodo.org/record/1175282
    # download and extract https://zenodo.org/record/1175282/files/TNBC_NucleiSegmentation.zip
    raw_input_dir = ROOT_DIR / 'raw_external/TNBC_NucleiSegmentation'

    raw_train_images_dir = raw_input_dir / 'images'
    raw_train_annotations_dir = raw_input_dir / 'masks'

    train_images_output_dir = DATASET_WORKING_DIR / dataset_name
    try:
        os.mkdir(train_images_output_dir.as_posix())
    except:
        pass

    annotations_file_path = DATASET_WORKING_DIR / 'annotations/TNBC_NucleiSegmentation.json'

    im_ids = image_ids_in(raw_train_images_dir)

    class_name_base = 'pink_tissue_TNBC_Nuclei'
    im_count = 0

    dataset_structure = {
        'annotations': [],
        'categories': [
            {
                'id': 1,
                'name': 'Nucleus'
            }
        ],
        'images': [],
        'info': {
            'description': 'coco format Nuclei external TNBC_NucleiSegmentation Dataset',
        }
    }

    for im_id in im_ids:
        im_metadata, annotations_res, im_count = load_image(raw_train_images_dir, raw_train_annotations_dir,
                                                            im_id, dataset_name, class_name_base,
                                                            im_count, train_images_output_dir)
        if not im_metadata:
            continue

        dataset_structure['images'].extend(im_metadata)
        dataset_structure['annotations'].extend(annotations_res)

        im_count += 1

    json.dump(dataset_structure, open(annotations_file_path.as_posix(), 'w'))


def prepare_BBBC006():
    dataset_name = 'BBBC006'

    # ref: https://data.broadinstitute.org/bbbc/image_sets.html
    # download this from the above link and extract
    raw_input_dir = ROOT_DIR / 'raw_external/BBBC006'

    raw_train_images_dir = raw_input_dir / 'images'
    raw_train_annotations_dir = raw_input_dir / 'masks'

    train_images_output_dir = DATASET_WORKING_DIR / dataset_name

    try:
        os.mkdir(train_images_output_dir.as_posix())
    except:
        pass

    annotations_file_path = DATASET_WORKING_DIR / 'annotations/BBBC006.json'

    im_ids = image_ids_in(raw_train_images_dir)

    images, rle_lists, image_sizes, train_im_ids = load_images(raw_train_images_dir,
                                                               raw_train_annotations_dir,
                                                               im_ids,
                                                               dataset=dataset_name)

    class_name_base = 'black_white_blur_BBBC006'

    im_count = 0
    dataset_structure = {
        'annotations': [],
        'categories': [
            {
                'id': 1,
                'name': 'Nucleus'
            }
        ],
        'images': [],
        'info': {
            'description': 'coco format Nuclei external BBBC006 Dataset',
        }
    }

    for im, im_id, masks, sz in zip(images, train_im_ids, rle_lists, image_sizes):
        im_metadata, annotations_res = get_image_data(im, masks, im_count,
                                                      im_id, sz,
                                                      '{}_{}'.format(class_name_base, np.random.choice([0, 1])),
                                                      train_images_output_dir)
        dataset_structure['images'].append(im_metadata)
        dataset_structure['annotations'].extend(annotations_res)

        im_count += 1

    json.dump(dataset_structure, open(annotations_file_path.as_posix(), 'w'))


def prepare_nucleisegmentationbenchmark():
    dataset_name = 'nucleisegmentationbenchmark'

    # ref: https://nucleisegmentationbenchmark.weebly.com/dataset.html
    # download and extract
    raw_input_dir = ROOT_DIR / 'raw_external/nucleisegmentationbenchmark'

    raw_train_images_dir = raw_input_dir / 'Tissue images'
    raw_train_annotations_dir = raw_input_dir / 'Annotations'

    train_images_output_dir = DATASET_WORKING_DIR / dataset_name

    try:
        os.mkdir(train_images_output_dir.as_posix())
    except:
        pass

    annotations_file_path = DATASET_WORKING_DIR / 'annotations/nucleisegmentationbenchmark.json'

    im_ids = image_ids_in(raw_train_images_dir)
    images, rle_lists, image_sizes = load_images_benchmark_tissue(raw_train_images_dir, im_ids,
                                                                                   raw_train_annotations_dir)

    image_tiles, rle_tile_list, image_tile_sizes, image_tile_names = preprocess_as_tiles(images, rle_lists, im_ids)

    im_count = 0
    dataset_structure = {
        'annotations': [],
        'categories': [
            {
                'id': 1,
                'name': 'Nucleus'
            }
        ],
        'images': [],
        'info': {
            'description': 'coco format Nuclei external nucleisegmentationbenchmark Dataset',
        }
    }

    classes = ['PurpleTissueBenchmarkType_{}'.format(i) for i in np.random.choice([0, 1], len(image_tiles))]

    for im, im_id, masks, sz, cls in zip(image_tiles, image_tile_names, rle_tile_list, image_tile_sizes, classes):
        im_metadata, annotations_res = get_image_data(im, masks, im_count, im_id, sz, cls,
                                                      train_images_output_dir)
        dataset_structure['images'].append(im_metadata)
        dataset_structure['annotations'].extend(annotations_res)
        im_count += 1

    json.dump(dataset_structure, open(annotations_file_path.as_posix(), 'w'))


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare external datasets')
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


def main():
    prepare_cluster_nuclei()
    prepare_2009_ISBI_2DNuclei_code_data()
    prepare_BBBC006()
    prepare_BBBC007()
    prepare_BBBC018()
    prepare_BBBC020()
    prepare_nuclei_partial_annotations()
    prepare_TNBC_NucleiSegmentation()
    prepare_nucleisegmentationbenchmark()


if __name__ == '__main__':
    args = parse_args()
    ROOT_DIR = Path(args.root_data_dir)
    main()
