import json
from pathlib import Path
import numpy as np
from PIL import Image
from pycocotools import mask as mask_util

ROOT_DIR = Path('/media/gangadhar/DataSSD1TB/ROOT_DATA_DIR/')
DATASET_WORKING_DIR = ROOT_DIR / 'Nuclei'

annotations_file = DATASET_WORKING_DIR / 'annotations/stage1_train.json'

COCO = json.load(open(annotations_file.as_posix()))

image_metadata = COCO['images'][0]
print image_metadata

# {u'file_name': u'4ca5081854df7bbcaa4934fcf34318f82733a0f8c05b942c2265eea75419d62f.jpg',
#  u'height': 256,
#  u'id': 0,
#  u'nuclei_class': u'purple_purple_320_256_sparce',
#  u'width': 320}


def get_masks(im_metadata):
    image_annotations = []
    for annotation in COCO['annotations']:
        if annotation['image_id'] == im_metadata['id']:
            image_annotations.append(annotation)

    segments = [annotation['segmentation'] for annotation in image_annotations]
    masks = mask_util.decode(segments)
    return masks


masks = get_masks(image_metadata)

print masks.shape
# (256, 320, 37)


def show(i):
    i = np.asarray(i, np.float)
    m,M = i.min(), i.max()
    I = np.asarray((i - m) / (M - m + 0.000001) * 255, np.uint8)
    Image.fromarray(I).show()


show(np.sum(masks, -1))
# this should show an image with all masks
