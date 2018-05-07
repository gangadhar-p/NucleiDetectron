# Nuclei Detectron

**10th (currently 7th place while leaderboard is being updated) place solution for the 2018 Data Science Bowl with a score of 0.591 with a single model submission.**

Nuclei Detectron is built on [Detectorn](https://github.com/facebookresearch/Detectron) which is a [Mask R-CNN](https://arxiv.org/abs/1703.06870) based solution. This approach was chosen since Mask R-CNN works well on instance segmentation problems.

_P.S. Unofficial score of 0.608 without ensemble on submissions that were not in the 2 final submissions._

## Resources related to this project

* [Data used to build the models](https://www.kaggle.com/gangadhar/nuclei-segmentation-in-microscope-cell-images)
* [Pretrained model, predictions and visualization of submission](https://www.kaggle.com/gangadhar/nuclei-detectron-models-for-2018-data-science-bowl)

## Introduction

The challenge is to build a model that can identify a range of nuclei across varied conditions.
The hard part of the challenge is to generalize across nuclei types that are very different from training set,
and in different lighting conditions.

### Samples from clusters of different nuclei used in training

<div align="center">
  <img src="demo/nuclei/clustering/clusters.png"/>
  <p></p>
</div>


### Example good predictions on Stage 2 test images:

| |  | |
:-------------------------:|:-------------------------: |:-------------------------:
![](demo/nuclei/sample_stage_2_predictions/success/3fceda40ce7dfc8129ab60f3e439de452f5e95f1004e98c7d9f8a53b89c1cfa4.png)  | ![](demo/nuclei/sample_stage_2_predictions/success/4f8089b39b27804a5a9471f0bd763fb3106e757368ebdfbf3c368e8a9bb69b98.png)  | ![](demo/nuclei/sample_stage_2_predictions/success/5e4cc5704b9660cc061d0e901de3f2d41c44a71286411aa12d650f25a73d36f1.png)
![](demo/nuclei/sample_stage_2_predictions/success/39b6ed3e0dd50ea6e113e4f95b656c9bdc6d3866579f648a16a853aeb5af1a61.png)  | ![](demo/nuclei/sample_stage_2_predictions/success/60fd9ffd0b8c95a4297504a48ade1f27797a53795be9fe6cb0e1ba2e71e6f606.png)  | ![](demo/nuclei/sample_stage_2_predictions/success/80a5d5b851304b761e7b035efda85de43ee9ce8d4593f93510c7940dc99dc219.png)
![](demo/nuclei/sample_stage_2_predictions/success/5390acefd575cf9b33413ddf6cbb9ce137ae07dc04616ba24c7b5fe476c827d2.png)  | ![](demo/nuclei/sample_stage_2_predictions/success/0682759f81d3e26c9cebc9973f297025bbdb07b419edf748d98bab84594bc2f1.png)  | ![](demo/nuclei/sample_stage_2_predictions/success/f0c40bcaa222468d2a32a13f6d5f145bb2fc9c18f408ddcf4e175594b398ab68.png)

### Example bad predictions on strange nuclei types in Stage 2 test image:

| |  | |
:-------------------------:|:-------------------------: |:-------------------------:
![](demo/nuclei/sample_stage_2_predictions/failure/43e02f592b3416fa74e606bde87b998b8c480bb2a73e0864ccc4860da2107c2a.png)  | ![](demo/nuclei/sample_stage_2_predictions/failure/064e6d7d49e155b12c10f3054a453a615714198cf9e83d9edf717c151b922c90.png)  | ![](demo/nuclei/sample_stage_2_predictions/failure/232b87a391ebb8ad19b0e755aa21267f08cc48df6ed2cc7aebb163ae4cd67909.png)
![](demo/nuclei/sample_stage_2_predictions/failure/7063ca81ca61a70754903315262507fc51d7d0bc9db29432376410afda222c2e.png)  | ![](demo/nuclei/sample_stage_2_predictions/failure/460671644f1c41abbc0e9659d3c15aa3f31a3135792ff849d0a40ca30874b589.png)  | ![](demo/nuclei/sample_stage_2_predictions/failure/a7730613067b597f6ae18202274fd08e855aef10998ba07b91062d6cf333d3c9.png)
![](demo/nuclei/sample_stage_2_predictions/failure/b87950af5f2f2c39b33f985b1b98df21ea0cdbade98df6346f0f8959dfdc60da.png)  | ![](demo/nuclei/sample_stage_2_predictions/failure/da44981210f3f498aa62b2e825889bfe3b896997b8dec1fd13d50830c63974ff.png)  | ![](demo/nuclei/sample_stage_2_predictions/failure/e1dedfc527eb4b9e0f85e3d9da0ec1f7343b9dfb7651df110958738ed831224b.png)


## Dataset preparation
* There were several nuclei datasets with outlines as annotations.
   * Applied classical computer vision techniques to convert ground truth from outlines to masks.
   * This involved adding boundary pixels to the image so all contours are closed.
   * Given outlines of cells with overlaps/touching or at border,
      * Mark an outer contour to encompass contours that are at image edges.
      * then do cv2.findContours to get the polygons of mask.
      * Ref [parse_segments_from_outlines](https://github.com/gangadhar-p/NucleiDetectron/blob/master/lib/datasets/nuclei/mask_encoding.py#L184)
* Standardized all datasets into COCO mask RLE JSON file format.
   * You can use [cocoapi](https://github.com/cocodataset/cocoapi) to load the annotations.
* Cut image into tiles when images are bigger than 1000 pixels
   * This was necessary since large image features did not fit in GPU memory.

## Preprocessing
* Cluster images into classes based on the color statistics.
* Normalize classes size
   * Oversample/undersample images from clusters to a constant number of images per class in each epoch.
* Fill holes in masks
* Split nuclei masks that are fused
   - Applied morphological Erosion and Dilation to seperate fused cells
   - Use statistics of nuclie sizes in an image to find outliers
*  [ZCA whitening of images](http://ufldl.stanford.edu/wiki/index.php/Whitening)
*  Zero mean unit variance normalization
*  Grey scale: [Color-to-Grayscale: Does the Method Matter in Image Recognition](http://tdlc.ucsd.edu/SV2013/Kanan_Cottrell_PLOS_Color_2012.pdf).
   - Very important how you convert to grey scale. Many algorithms for the conversion, loss of potential data.
   - Luminous
   - Intensity
   - Value: This is the method I used.
*  [Contrast Limited Adaptive Histogram Equalization](https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html)

## Augmentation

Data augmentation is one of the key to achieve good generalization in this challenge.

### Training time augmentation

* Invert
  * This augmentation helped in reducing generalization error significantly
  * Randomly choosing to invert caused the models to generalize across all kids of backgrounds in the local validation set.
* Geometric
  * PerspectiveTransform
    * This is very useful to make the circular looking cells to look stretched
  * PiecewiseAffine
  * Flip
  * Rotate (0, 90, 180, 270)
  * Crop
* Alpha blending
  * Create geometrical blur by affine operation
  * Shear, rotate, translate, scale
* Pixel
  * AddToHueAndSaturation
  * Multiply
  * Dropout, CoarseDropout
  * ContrastNormalization
* Noise
  * AdditiveGaussianNoise
  * SimplexNoiseAlpha
  * FrequencyNoiseAlpha
* Blur
  * GaussianBlur
  * AverageBlur
  * MedianBlur
  * BilateralBlur
* Texture
  * Superpixels
  * Sharpen
  * Emboss
  * EdgeDetect
  * DirectedEdgeDetect
  * ElasticTransformation


| Original Image | Random Augmentations for training |
:-------------------------:|:-------------------------:
![](demo/nuclei/augmentation/1b44d22643830cd4f23c9deadb0bd499fb392fb2cd9526d81547d93077d983df.jpg)  | ![](demo/nuclei/augmentation/Augmentation4.jpg)
![](demo/nuclei/augmentation/4d09672bcf5a2661eea00891bbb8191225a06619a849aece37ad10d9dedbde3e.jpg)  | ![](demo/nuclei/augmentation/Augmentation2.jpg)

## Color Transfer Results

| Source Image | Target Image color style | Result Color Style |
:-------------------------:|:-------------------------: |:-------------------------:
![](demo/nuclei/transfer/a.png)  | ![](demo/nuclei/transfer/b.jpg)  | ![](demo/nuclei/transfer/c.jpg)


| | |
:-------------------------:|:-------------------------:
![](demo/nuclei/transfer/TCGA-49-4488-01Z-00-DX1.tifbb3e626a-1559-4890-9ee0-5ac4b1da48ba.jpg)  | ![](demo/nuclei/transfer/TCGA-49-4488-01Z-00-DX1.tifd9ca395e-e16b-4e3e-9932-2c459b53bd87.jpg)
![](demo/nuclei/transfer/TCGA-49-4488-01Z-00-DX1.tifd59e8744-99b0-4cc7-97cc-147c63d193a7.jpg)   | ![](demo/nuclei/transfer/TCGA-49-4488-01Z-00-DX1.tife21ccb29-785a-4e18-8a4a-103258b338d3.jpg)

### Test time augmentation
1. Invert: Have improved the performance a lot
2. Multiple Scales 900, 1000, 1100
3. Flip left right


## Architecture changes to baseline Detectron

Detectron network configuration changes from the baseline e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x.yaml are:

1. Create small anchor sizes for small nuclei. RPN_ANCHOR_START_SIZE: 8 # default 32
2. Add more aspect rations for nuclei that are close but in cylindrical structure. RPN_ASPECT_RATIOS: (0.2, 0.5, 1, 2, 5)
3. Increase the ROI resolution. ROI_XFORM_RESOLUTION: 14
4. Increase the number of detections per image from default 100. DETECTIONS_PER_IM: 500

## Training
1. Decreased warmup fraction to 0.01
2. Increased warmup iterations to 10,000
3. Gave mask loss more weight WEIGHT_LOSS_MASK: 1.2

## Segmentation Post processing
  * Threshold on area to remove masks below area of 15 pixels
  * Threshold on BBox confidence of 0.9
  * Mask NMS
    * On decreasing order of confidence, simple union-mask strategy to remove overlapping segments or cut segments at overlaps if overlap is below 30% of the mask.

## What worked most
1. Inversion in augmentation
2. Blurring and frequency noise
3. Additional datasets, even though they caused a drop on the public leaderboard, I noticed no drop in local validation set.

## What did not work
1. Mask dilations and erosions
   * This did not have any improvement in the segmentation in my experiments
2. Use contour approximations in place of original masks
   * This did not have any improvement either. Maybe this could add a boost if using light augmentations.
3. Randomly apply structuring like open-close
4. Soft NMS thresh
   * Did not improve accuracy
5. Color images
   * Did not perform as well as grey images after augmentations
6. Color style transfer. Take a source image and apply the color style to target image.
7. Style transfer: Was losing a lot of details on some nuclei but looked good on very few images.
8. Dilation of masks in post processing, this drastically increased error because the model masks are already good.
9. Distance transform and split masks during training.

## Things I didn't have time to try
1. Ensemble multiple Mask R-CNN's
2. Two stage predictions with U-Net after box proposals.
3. Augmentation smoothing during training
   * Increase the noise and augmentation slowly during the training phase, like from 10% to 50%
   * Reduce the augmentation from 90% to 20% during training, for generalization and fitting.
4. Experiment with different levels of augmentation individually across, noise, blur, texture, alpha blending.
5. Different layer normalization techniques, with batch size more than one image at a time. Need bigger GPU.
6. Little bit of hyperparameter search on thresholds and network architecture.

## Things I did not think of
U-Net with watershed, did not think this approach would outperform Mask R-CNN


# Installation
For basic host setup of Nvidia driver and Nvidia-Docker go to [`setup.sh`](bin/nuclei/setup.sh).
Please find installation instructions for Caffe2 and Detectron in [`INSTALL.md`](INSTALL.md).


## Training and testing and look at logs
```bash
chmod +x bin/nuclei/train.sh && ./bin/nuclei/train.sh -e 1_aug_gray_0_5_0 -v 1_aug_gray_0_5_0 -g 1 &

chmod +x bin/nuclei/test.sh && ./bin/nuclei/test.sh -e 1_aug_gray_1_5_1_stage_2_v1 -v 1_aug_gray_1_5_1_stage_2_v1 -g 1 &

tail -f /detectron/lib/datasets/data/logs/test_log

python lib/datasets/nuclei/write_submission.py \
    --results-root /detectron/lib/datasets/data/results/ \
    --run-version '1_aug_gray_1_5_1_stage_2_v1' \
    --iters '65999' \
    --area-thresh 15 \
    --acc-thresh 0.9 \
    --intersection-thresh 0.3

```


## Code References
- [Detectron](https://github.com/facebookresearch/detectron).
  Ross Girshick and Ilija Radosavovic. Georgia Gkioxari. Piotr Doll\'{a}r. Kaiming He.
  Github, Jan. 2018.

- [Image augmentation for machine learning experiments](https://github.com/aleju/imgaug).
  Alexander Jung.
  Github, Jan. 2015.

- [Normalizing brightfield, stained and fluorescence](https://www.kaggle.com/kmader/normalizing-brightfield-stained-and-fluorescence).
  Kevin Mader.
  Kaggle Notebook, Apr. 2018.

- [Fast, tested RLE and input routines](https://www.kaggle.com/stainsby/fast-tested-rle-and-input-routines).
  Sam Stainsby.
  Kaggle Notebook, Apr. 2018.

- [Example Metric Implementation](https://www.kaggle.com/wcukierski/example-metric-implementation).
  William Cukierski.
  Kaggle Notebook, Apr. 2018.


# Datasets

A collection of datasets converted into COCO segmentation format.

## Preprocessing:
 - Resized few images
 - Tiled some images with lot of annotations to fit in memory
 - Extracted masks when only outlines were available
   - This is done by finding contours

## Folder hierarchy

```python

DATASETS = {
    'nuclei_stage1_train': {
        IM_DIR:
            _DATA_DIR + '/Nuclei/stage_1_train',
        ANN_FN:
            _DATA_DIR + '/Nuclei/annotations/stage1_train.json'
    },
    'nuclei_stage_1_local_train_split': {
        IM_DIR:
            _DATA_DIR + '/Nuclei/stage_1_train',
        ANN_FN:
            _DATA_DIR + '/Nuclei/annotations/stage_1_local_train_split.json'
    },
    'nuclei_stage_1_local_val_split': {
        IM_DIR:
            _DATA_DIR + '/Nuclei/stage_1_train',
        ANN_FN:
            _DATA_DIR + '/Nuclei/annotations/stage_1_local_val_split.json'
    },
    'nuclei_stage_1_test': {
        IM_DIR:
            _DATA_DIR + '/Nuclei/stage_1_test',
        ANN_FN:
            _DATA_DIR + '/Nuclei/annotations/stage_1_test.json'
    },
    'nuclei_stage_2_test': {
        IM_DIR:
            _DATA_DIR + '/Nuclei/stage_2_test',
        ANN_FN:
            _DATA_DIR + '/Nuclei/annotations/stage_2_test.json'
    },
    'cluster_nuclei': {
        IM_DIR:
            _DATA_DIR + '/Nuclei/cluster_nuclei',
        ANN_FN:
            _DATA_DIR + '/Nuclei/annotations/cluster_nuclei.json'
    },
    'BBBC007': {
        IM_DIR:
            _DATA_DIR + '/Nuclei/BBBC007',
        ANN_FN:
            _DATA_DIR + '/Nuclei/annotations/BBBC007.json'
    },
    'BBBC006': {
        IM_DIR:
            _DATA_DIR + '/Nuclei/BBBC006',
        ANN_FN:
            _DATA_DIR + '/Nuclei/annotations/BBBC006.json'
    },
    'BBBC018': {
        IM_DIR:
            _DATA_DIR + '/Nuclei/BBBC018',
        ANN_FN:
            _DATA_DIR + '/Nuclei/annotations/BBBC018.json'
    },
    'BBBC020': {
        IM_DIR:
            _DATA_DIR + '/Nuclei/BBBC020',
        ANN_FN:
            _DATA_DIR + '/Nuclei/annotations/BBBC020.json'
    },
    'nucleisegmentationbenchmark': {
        IM_DIR:
            _DATA_DIR + '/Nuclei/nucleisegmentationbenchmark',
        ANN_FN:
            _DATA_DIR + '/Nuclei/annotations/nucleisegmentationbenchmark.json'
    },
    '2009_ISBI_2DNuclei': {
        IM_DIR:
            _DATA_DIR + '/Nuclei/2009_ISBI_2DNuclei',
        ANN_FN:
            _DATA_DIR + '/Nuclei/annotations/2009_ISBI_2DNuclei.json'
    },
    'nuclei_partial_annotations': {
        IM_DIR:
            _DATA_DIR + '/Nuclei/nuclei_partial_annotations',
        ANN_FN:
            _DATA_DIR + '/Nuclei/annotations/nuclei_partial_annotations.json'
    },
    'TNBC_NucleiSegmentation': {
        IM_DIR:
            _DATA_DIR + '/Nuclei/TNBC_NucleiSegmentation',
        ANN_FN:
            _DATA_DIR + '/Nuclei/annotations/TNBC_NucleiSegmentation.json'
    },
}
```

## Example usage:

```python

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

```

## Data References

- [2018 Data Science Bowl: Find the nuclei in divergent images to advance medical discovery](https://www.kaggle.com/c/data-science-bowl-2018).
  Competition, Kaggle, Apr. 2018.
  Download: https://www.kaggle.com/c/data-science-bowl-2018/data

- [2018 Data Science Bowl: Kaggle Data Science Bowl 2018 dataset fixes](https://github.com/lopuhin/kaggle-dsbowl-2018-dataset-fixes).
  Konstantin Lopuhin, Apr. 2018.
  Download: https://github.com/lopuhin/kaggle-dsbowl-2018-dataset-fixes

- [TNBC_NucleiSegmentation: A dataset for nuclei segmentation based on Breast Cancer patients](https://zenodo.org/record/1175282).
  Naylor Peter Jack; Walter Thomas; Laé Marick; Reyal Fabien. 2018.
  Download: https://zenodo.org/record/1175282/files/TNBC_NucleiSegmentation.zip

- [A Dataset and a Technique for Generalized Nuclear Segmentation for Computational Pathology](https://www.ncbi.nlm.nih.gov/pubmed/28287963).
  Kumar N, Verma R, Sharma S, Bhargava S, Vahadane A, Sethi A. 2017.
  Download: https://nucleisegmentationbenchmark.weebly.com/dataset.html

- [Deep learning for digital pathology image analysis: A comprehensive tutorial with selected use cases](http://www.jpathinformatics.org/article.asp?issn=2153-3539;year=2016;volume=7;issue=1;spage=29;epage=29;aulast=Janowczyk).
  Andrew Janowczyk, Anant Madabhushi. 2016.
  Download: http://andrewjanowczyk.com/wp-static/nuclei.tgz

- [Nuclei Dataset: Include 52 images of 200x200 pixels](https://imagej.nih.gov/ij/plugins/ihc-toolbox/index.html).
  Jie Shu, Guoping Qiu, Mohammad Ilyas.
  Immunohistochemistry (IHC) Image Analysis Toolbox, Jan. 2015.
  Download: https://www.dropbox.com/s/9knzkp9g9xt6ipb/cluster%20nuclei.zip?dl=0

- [BBBC006v1: image set available from the Broad Bioimage Benchmark Collection](https://data.broadinstitute.org/bbbc/BBBC007/).
  Vebjorn Ljosa, Katherine L Sokolnicki & Anne E Carpenter. 2012.
  Download: https://data.broadinstitute.org/bbbc/BBBC007/

- [BBBC007v1: image set available from the Broad Bioimage Benchmark Collection](https://data.broadinstitute.org/bbbc/BBBC006/).
  Vebjorn Ljosa, Katherine L Sokolnicki & Anne E Carpenter. 2012.
  Download: https://data.broadinstitute.org/bbbc/BBBC006/

- [BBBC018v1: image set available from the Broad Bioimage Benchmark Collection](https://data.broadinstitute.org/bbbc/BBBC018/).
  Vebjorn Ljosa, Katherine L Sokolnicki & Anne E Carpenter. 2012.
  Download: https://data.broadinstitute.org/bbbc/BBBC018/

- [BBBC020v1: image set available from the Broad Bioimage Benchmark Collection](https://data.broadinstitute.org/bbbc/BBBC020/).
  Vebjorn Ljosa, Katherine L Sokolnicki & Anne E Carpenter. 2012.
  Download: https://data.broadinstitute.org/bbbc/BBBC020/

- [Nuclei Segmentation In Microscope Cell Images: A Hand-Segmented Dataset And Comparison Of Algorithms](http://murphylab.web.cmu.edu/data/2009_ISBI_Nuclei.html).
  L. P. Coelho, A. Shariff, and R. F. Murphy.
  Proceedings of the 2009 IEEE International Symposium on Biomedical Imaging (ISBI 2009), pp. 518-521, 2009.
  Download: http://murphylab.web.cmu.edu/data/2009_ISBI_2DNuclei_code_data.tgz

