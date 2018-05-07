import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap
from core.config import cfg, get_worker_seed
from datasets.nuclei.mask_morphology import masks_augmentation
import utils.boxes as box_utils
import multiprocessing

ia.seed(get_worker_seed())

half_time_prob = 0.5
one_third_times = 0.333
infrequent = 0.2
rare_probable = 0.1

half_times = lambda aug: iaa.Sometimes(half_time_prob, aug)
sometimes = lambda aug: iaa.Sometimes(one_third_times, aug)
fewtimes = lambda aug: iaa.Sometimes(infrequent, aug)
rarely = lambda aug: iaa.Sometimes(rare_probable, aug)
unlikely = lambda aug: iaa.Sometimes(0.05, aug)


def black_and_white_aug():
    alpha_seconds = iaa.OneOf([
        iaa.Affine(rotate=(-3, 3)),
        iaa.Affine(translate_percent={"x": (0.95, 1.05), "y": (0.95, 1.05)}),
        iaa.Affine(scale={"x": (0.95, 1.05), "y": (0.95, 1.05)}),
        iaa.Affine(shear=(-2, 2)),
        iaa.CoarseDropout(p=0.1, size_percent=(0.08, 0.02)),
    ])

    first_set = iaa.OneOf([
        iaa.Multiply(iap.Choice([0.5, 1.5]), per_channel=True),
        iaa.EdgeDetect((0.1, 1)),
    ])

    second_set = iaa.OneOf(
        [
            iaa.AddToHueAndSaturation((-40, 40)),
            iaa.ContrastNormalization((0.5, 2.0), per_channel=True)
        ]
    )

    color_aug = iaa.Sequential(
        [
            # Original Image Domain ==================================================

            # Geometric Rigid
            iaa.Fliplr(0.5),
            iaa.OneOf([
                iaa.Noop(),
                iaa.Affine(rotate=90),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270),
            ]),

            iaa.OneOf([
                iaa.Noop(),
                iaa.Crop(percent=(0, 0.1)),  # Random Crops
                iaa.PerspectiveTransform(scale=(0.05, 0.15)),
            ]),

            # Affine
            sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.07), nb_rows=(3, 6), nb_cols=(3, 6))),
            fewtimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-45, 45),
                shear=(-16, 16),
                order=[0, 1],
                cval=0)
            ),

            # Transformations outside Image domain ==============================================

            # COLOR, CONTRAST, HUE
            iaa.Invert(0.5, name='Invert'),
            fewtimes(iaa.Add((-10, 10), per_channel=0.5, name='Add')),
            fewtimes(iaa.AddToHueAndSaturation((-40, 40), per_channel=0.5, name='AddToHueAndSaturation')),

            # Intensity / contrast
            fewtimes(iaa.ContrastNormalization((0.8, 1.1), name='ContrastNormalization')),

            # Add to hue and saturation
            fewtimes(iaa.Multiply((0.5, 1.5), per_channel=0.5, name='HueAndSaturation')),

            # Noise ===========================================================================
            fewtimes(iaa.AdditiveGaussianNoise(loc=0,
                                               scale=(0.0, 0.15 * 255), per_channel=0.5,
                                               name='AdditiveGaussianNoise')),

            fewtimes(
                iaa.Alpha(
                    factor=(0.5, 1),
                    first=iaa.ContrastNormalization((0.5, 2.0), per_channel=True),
                    second=alpha_seconds,
                    per_channel=0.5,
                    name='AlphaNoise'
                ),
            ),
            fewtimes(
                iaa.SimplexNoiseAlpha(
                    first=first_set,
                    second=second_set,
                    per_channel=0.5,
                    aggregation_method="max",
                    sigmoid=False,
                    upscale_method='cubic',
                    size_px_max=(2, 12),
                    name='SimplexNoiseAlpha'
                ),
            ),
            fewtimes(
                iaa.FrequencyNoiseAlpha(
                    first=first_set,
                    second=second_set,
                    per_channel=0.5,
                    aggregation_method="max",
                    sigmoid=False,
                    upscale_method='cubic',
                    size_px_max=(2, 12),
                    name='FrequencyNoiseAlpha'
                ),
            ),

            # Blur
            fewtimes(iaa.OneOf([
                iaa.GaussianBlur((0, 3.0)),
                iaa.AverageBlur(k=(2, 7)),
                iaa.MedianBlur(k=(3, 11)),
                iaa.BilateralBlur(d=(3, 10), sigma_color=(10, 250), sigma_space=(10, 250))
            ], name='Blur')),

            # Regularization ======================================================================
            unlikely(iaa.OneOf(
                [
                    iaa.Dropout((0.01, 0.1), per_channel=0.5, name='Dropout'),
                    iaa.CoarseDropout(
                        (0.03, 0.15), size_percent=(0.02, 0.05),
                        per_channel=0.5,
                        name='CoarseDropout'
                    ),
                ],

            )),

        ],
        random_order=True)

    seq = iaa.Sequential(
        [
            iaa.Sequential(
                [
                    # Texture
                    rarely(iaa.Superpixels(p_replace=(0.3, 1.0), n_segments=(500, 1000), name='Superpixels')),
                    rarely(iaa.Sharpen(alpha=(0, 0.5), lightness=(0.75, 1.0), name='Sharpen')),
                    rarely(iaa.Emboss(alpha=(0, 1.0), strength=(0, 1.0), name='Emboss')),
                    rarely(iaa.OneOf([
                        iaa.EdgeDetect(alpha=(0, 0.5)),
                        iaa.DirectedEdgeDetect(
                            alpha=(0, 0.5), direction=(0.0, 1.0)
                        ),
                    ], name='EdgeDetect')),
                    rarely(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25, name='ElasticTransformation')),
                ],
                random_order=True
            ),

            color_aug,

            iaa.Grayscale(alpha=1.0, name='Grayscale')
        ], random_order=False

    )

    def activator_masks(images, augmenter, parents, default):
        if 'Unnamed' not in augmenter.name:
            return False
        else:
            return default

    hooks_masks = ia.HooksImages(activator=activator_masks)
    return seq, hooks_masks


seq_old, hooks_masks_old = black_and_white_aug()


def get_optimistic_img_aug():
    texture = iaa.OneOf([
        iaa.Superpixels(p_replace=(0.1, 0.3), n_segments=(500, 1000),
                        interpolation="cubic", name='Superpixels'),
        iaa.Sharpen(alpha=(0, 1.0), lightness=(0.5, 1.0), name='Sharpen'),
        iaa.Emboss(alpha=(0, 1.0), strength=(0.1, 0.3), name='Emboss'),
        iaa.OneOf([
            iaa.EdgeDetect(alpha=(0, 0.4)),
            iaa.DirectedEdgeDetect(
                alpha=(0, 0.7), direction=(0.0, 1.0)
            ),
        ], name='EdgeDetect'),
        iaa.ElasticTransformation(alpha=(0.5, 1.0), sigma=0.2, name='ElasticTransformation'),
    ])

    blur = iaa.OneOf([
        iaa.GaussianBlur((1, 5.0), name='GaussianBlur'),
        iaa.AverageBlur(k=(2, 15), name='AverageBlur'),
        iaa.MedianBlur(k=(3, 15), name='MedianBlur'),
        iaa.BilateralBlur(d=(3, 15), sigma_color=(10, 250), sigma_space=(10, 250),
                          name='BilaBlur'),
    ])

    affine = iaa.OneOf(
        [
            iaa.Affine(rotate=(-3, 3)),
            iaa.Affine(translate_percent={"x": (0.95, 1.05), "y": (0.95, 1.05)}),
            iaa.Affine(scale={"x": (0.95, 1.05), "y": (0.95, 1.05)}),
            iaa.Affine(shear=(-2, 2)),
        ]
    )

    factors = iaa.OneOf([
        iaa.Multiply(iap.Choice([0.75, 1.25]), per_channel=False),
        iaa.EdgeDetect(1.0),
    ])

    seq = iaa.Sequential(
        [

            # Size and shape ==================================================

            iaa.Sequential([

                iaa.Fliplr(0.5),

                iaa.OneOf([
                    iaa.Noop(),
                    iaa.Affine(rotate=90),
                    iaa.Affine(rotate=180),
                    iaa.Affine(rotate=270),
                ]),

                half_times(iaa.SomeOf((1, 2),
                           [
                               iaa.Crop(percent=(0.1, 0.4)),  # Random Crops
                               iaa.PerspectiveTransform(scale=(0.10, 0.175)),
                               iaa.PiecewiseAffine(scale=(0.01, 0.06), nb_rows=(3, 6), nb_cols=(3, 6)),
                           ])),
            ]),

            # Texture ==================================================

            sometimes(iaa.SomeOf((1, 2),
                       [
                           texture,
                           iaa.Alpha(
                               (0.0, 1.0),
                               first=texture,
                               per_channel=False
                           )
                       ], random_order=True,
                       name='Texture')),

            half_times(iaa.SomeOf((1, 2),
                                  [
                                      blur,
                                      iaa.Alpha(
                                          (0.0, 1.0),
                                          first=blur,
                                          per_channel=False
                                      ),

                                      iaa.Alpha(
                                          factor=(0.2, 0.8),
                                          first=iaa.Sequential([
                                              affine,
                                              blur,
                                          ]),
                                          per_channel=False
                                      ),
                                  ], random_order=True,
                                  name='Blur')),
            # Noise ==================================================

            sometimes(iaa.SomeOf((1, 2),
                       [
                           # Just noise
                           iaa.AdditiveGaussianNoise(loc=0,
                                                     scale=(0.0, 0.15 * 255),
                                                     per_channel=False,
                                                     name='AdditiveGaussianNoise'),

                           iaa.SaltAndPepper(0.05, per_channel=False, name='SaltAndPepper'),

                           # Regularization

                           iaa.Dropout((0.01, 0.1), per_channel=False, name='Dropout'),
                           iaa.CoarseDropout(
                               (0.03, 0.15), size_percent=(0.02, 0.05),
                               per_channel=False,
                               name='CoarseDropout'
                           ),

                           iaa.Alpha(
                               factor=(0.2, 0.8),
                               first=texture,
                               second=iaa.CoarseDropout(p=0.1, size_percent=(0.02, 0.05)),
                               per_channel=False,
                           ),

                           # Perlin style noise

                           iaa.SimplexNoiseAlpha(
                               first=factors,
                               per_channel=False,
                               aggregation_method="max",
                               sigmoid=False,
                               upscale_method='cubic',
                               size_px_max=(2, 12),
                               name='SimplexNoiseAlpha'
                           ),

                           iaa.FrequencyNoiseAlpha(
                               first=factors,
                               per_channel=False,
                               aggregation_method="max",
                               sigmoid=False,
                               upscale_method='cubic',
                               size_px_max=(2, 12),
                               name='FrequencyNoiseAlpha'
                           ),

                       ], random_order=True,
                       name='Noise'
                       )),

        ], random_order=False
    )

    def activator_masks(images, augmenter, parents, default):
        if 'Unnamed' not in augmenter.name:
            return False
        else:
            return default

    hooks_masks = ia.HooksImages(activator=activator_masks)

    return seq, hooks_masks


seq_optimistic, hooks_optimistic_masks = get_optimistic_img_aug()


def show_img(i):
    from PIL import Image
    import numpy as np
    i = np.asarray(i, np.float)
    m, M = i.min(), i.max()
    I = np.asarray((i - m) / (M - m) * 255, np.uint8)
    Image.fromarray(I).show()


def show_masks(masks, dim=-1):
    import numpy as np
    M1 = np.sum(masks, axis=dim)
    M1[M1 > 0] = 255
    show_img(M1)


def scale_boxes(boxes):
    import numpy as np

    x_fac = np.random.normal(loc=0.0, scale=1) / 50

    boxes[:, 0] = boxes[:, 0] + (boxes[:, 2] * x_fac)
    boxes[:, 2] = boxes[:, 2] - (boxes[:, 2] * x_fac * 2)

    y_fac = np.random.normal(loc=0.0, scale=1) / 50

    boxes[:, 1] = boxes[:, 1] + (boxes[:, 3] * y_fac)
    boxes[:, 3] = boxes[:, 3] - (boxes[:, 3] * y_fac * 2)
    return boxes


def augment_images(roidb, augment_masks=False, augment_boxes=False):
    from pycocotools import mask as mask_util
    import numpy as np
    from scipy import ndimage

    if cfg.TRAIN.AUGMENTATION_MODE == 'OPTIMISTIC':
        seq, hooks_masks = seq_optimistic, hooks_optimistic_masks
    else:
        seq, hooks_masks = seq_old, hooks_masks_old

    import cv2
    ims = [cv2.imread(roi['image'])[:, :, [2, 1, 0]] for roi in roidb]
    seq_det = seq.to_deterministic()  # call this for each batch again, NOT only once at the start

    aug_ims_augs = seq_det.augment_images(ims)

    orig_masks = [mask_util.decode(roi['segms']) for roi in roidb]

    if augment_masks and np.random.random() > 0.5:
        orig_masks = masks_augmentation(orig_masks)

    mask_augs = seq_det.augment_images(orig_masks, hooks=hooks_masks)

    for idx in range(mask_augs[0].shape[2]):
        mask_augs[0][:, :, idx] = ndimage.morphology.binary_fill_holes(mask_augs[0][:, :, idx]).astype(np.uint8)

    im_augs, roi_augs = [], []

    for im, im_aug, roi, M in zip(ims, aug_ims_augs, roidb, mask_augs):
        M[M > 0] = 1
        aug_rles = mask_util.encode(np.asarray(M, order='F'))

        valid = np.sum(M, axis=(0, 1)) > 15
        aug_rles = [rle for idx, rle in enumerate(aug_rles) if valid[idx]]

        n_masks = len(aug_rles)
        if aug_rles:
            for rle_aug in aug_rles:
                rle_aug['size'] = [int(i) for i in rle_aug['size']]
            new_boxes = np.float32(mask_util.toBbox(aug_rles))

            if augment_boxes and np.random.random() > 0.5:
                new_boxes = scale_boxes(new_boxes)

            new_boxes = box_utils.xywh_to_xyxy(new_boxes)
            roi_aug = roi.copy()
            roi_aug['box_to_gt_ind_map'] = np.asarray(range(n_masks), np.int32)  # roi['box_to_gt_ind_map'][:n_masks]  #
            roi_aug['boxes'] = new_boxes
            roi_aug['gt_classes'] = roi['gt_classes'][:n_masks]
            roi_aug['gt_overlaps'] = roi['gt_overlaps'][:n_masks]  # csr_matrix(roi['gt_overlaps'].todense()[:n_masks])
            roi_aug['is_crowd'] = roi['is_crowd'][:n_masks]
            roi_aug['max_classes'] = roi['max_classes'][:n_masks]
            roi_aug['max_overlaps'] = roi['max_overlaps'][:n_masks]
            roi_aug['seg_areas'] = np.float32(mask_util.area(aug_rles))
            roi_aug['segms'] = aug_rles

            roi_augs.append(roi_aug)
            im_augs.append(im_aug)
        else:
            roi_augs.append(roi)
            im_augs.append(im)

    im_augs = [im[:, :, [2, 1, 0]] for im in im_augs]

    return roi_augs, im_augs
