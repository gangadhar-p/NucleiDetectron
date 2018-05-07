# borrowed from https://www.kaggle.com/kmader/normalizing-brightfield-stained-and-fluorescence
import imageio
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from glob import glob
import os
from skimage.io import imread
import matplotlib.pyplot as plt
import seaborn as sns
from datasets.json_dataset import JsonDataset
from pycocotools import mask as mask_util
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from pathlib import Path


NUM_CLUSTERS = 30
TOL = 0.00000001

ROOT_DIR = Path('/media/gangadhar/DataSSD1TB/ROOT_DATA_DIR/')
DATASET_WORKING_DIR = ROOT_DIR / 'Nuclei'

ds = ('nuclei_stage_1_local_train_split',
      'nuclei_stage_1_local_val_split',
      # 'nuclei_stage_1_test',
      'nucleisegmentationbenchmark',
      'cluster_nuclei',
      'BBBC007',
      'BBBC018',
      'BBBC020',
      '2009_ISBI_2DNuclei',
      'nuclei_partial_annotations',
      'TNBC_NucleiSegmentation',
      )


color_features_names = ['Gray', 'Red', 'Green', 'Blue', 'Red-Green', 'Red-Green-Sd', 'Cell_Red',
                        'Cell_Green', 'Cell_Blue', 'Cell_Gray', 'Cell_Red-Green', 'Cell_Red-Green-Sd',
                        'Bg_Red', 'Bg_Green', 'Bg_Blue', 'Bg_Gray', 'Bg_Red-Green', 'Bg_Red-Green-Sd']


def load_images():
    all_roi = []
    for d in ds:
        all_roi.append(JsonDataset(d).get_roidb(gt=True))

    roidb = {}

    for a in all_roi:
        for roi in a:
            fname = roi['image'].rsplit('/', 1)[1]
            roidb[fname] = roi

    for roi in roidb.values():
        M = mask_util.decode(roi['segms'])
        M = np.sum(M, -1)
        roi['M'] = M

    all_images = glob(os.path.join(DATASET_WORKING_DIR.as_posix(), 'images_train_fixed', '*.jpg'))

    img_df = pd.DataFrame({'path': all_images})
    img_id = lambda in_path: in_path.rsplit('/', 1)[1]
    img_group = lambda in_path: 'Train'

    img_df['ImageId'] = img_df['path'].map(img_id)
    img_df['TrainingSplit'] = img_df['ImageId'].map(img_group)

    img_df.sample(2)

    img_df['images'] = img_df['path'].map(imread)
    img_df.drop(['path'], 1, inplace=True)
    img_df.sample(1)
    return img_df, roidb


def create_color_features(in_df, roidb):
    in_df['Red'] = in_df['images'].map(lambda x: np.mean(x[:, :, 0]))
    in_df['Green'] = in_df['images'].map(lambda x: np.mean(x[:, :, 1]))
    in_df['Blue'] = in_df['images'].map(lambda x: np.mean(x[:, :, 2]))
    in_df['Gray'] = in_df['images'].map(lambda x: np.mean(x))
    in_df['Red-Green'] = in_df['images'].map(lambda x: np.mean(x[:, :, 0] - x[:, :, 1]))
    in_df['Red-Green-Sd'] = in_df['images'].map(lambda x: np.std(x[:, :, 0] - x[:, :, 1]))

    in_df['Cell_Red'] = in_df.apply(lambda x: np.mean(roidb[x['ImageId']]['M'] > 0 * x['images'][:, :, 0]), axis=1)
    in_df['Cell_Green'] = in_df.apply(lambda x: np.mean(roidb[x['ImageId']]['M'] > 0 * x['images'][:, :, 1]), axis=1)
    in_df['Cell_Blue'] = in_df.apply(lambda x: np.mean(roidb[x['ImageId']]['M'] > 0 * x['images'][:, :, 2]), axis=1)
    in_df['Cell_Gray'] = in_df.apply(lambda x: np.mean(roidb[x['ImageId']]['M'] > 0 * x['images'][:, :, 0]), axis=1)
    in_df['Cell_Red-Green'] = in_df.apply(
        lambda x: np.mean(roidb[x['ImageId']]['M'] > 0 * (x['images'][:, :, 0] - x['images'][:, :, 1])), axis=1)
    in_df['Cell_Red-Green-Sd'] = in_df.apply(
        lambda x: np.std(roidb[x['ImageId']]['M'] > 0 * (x['images'][:, :, 0] - x['images'][:, :, 1])), axis=1)

    in_df['Bg_Red'] = in_df.apply(lambda x: np.mean(roidb[x['ImageId']]['M'] == 0 * x['images'][:, :, 0]), axis=1)
    in_df['Bg_Green'] = in_df.apply(lambda x: np.mean(roidb[x['ImageId']]['M'] == 0 * x['images'][:, :, 1]), axis=1)
    in_df['Bg_Blue'] = in_df.apply(lambda x: np.mean(roidb[x['ImageId']]['M'] == 0 * x['images'][:, :, 2]), axis=1)
    in_df['Bg_Gray'] = in_df.apply(lambda x: np.mean(roidb[x['ImageId']]['M'] == 0 * x['images'][:, :, 0]), axis=1)
    in_df['Bg_Red-Green'] = in_df.apply(
        lambda x: np.mean(roidb[x['ImageId']]['M'] == 0 * (x['images'][:, :, 0] - x['images'][:, :, 1])), axis=1)
    in_df['Bg_Red-Green-Sd'] = in_df.apply(
        lambda x: np.std(roidb[x['ImageId']]['M'] == 0 * (x['images'][:, :, 0] - x['images'][:, :, 1])), axis=1)

    return in_df


def create_color_cluster_kmeans(in_df, cluster_count=3):
    cluster_maker = KMeans(cluster_count)

    cluster_maker.fit_predict(in_df[color_features_names])

    in_df['cluster-id'] = in_df['cluster-id'].map(lambda x: str(x))
    return in_df


def create_color_cluster_DBSCAN(in_df, cluster_count=3):
    cluster_maker = DBSCAN(eps=0.3, min_samples=cluster_count)

    cluster_maker.fit_predict(in_df[color_features_names])

    in_df['cluster-id'] = in_df['cluster-id'].map(lambda x: str(x))
    return in_df


def create_color_cluster_agglomerative_clustering(in_df, num_clusters):
    cluster_maker = AgglomerativeClustering(linkage='average', n_clusters=num_clusters)

    cluster_maker.fit(in_df[color_features_names])

    in_df['cluster-id'] = cluster_maker.labels_

    in_df['cluster-id'] = in_df['cluster-id'].map(lambda x: str(x))
    return in_df


def visualize_cluster_samples(img_df, n_samples=6):
    grouper = img_df.groupby(['cluster-id', 'TrainingSplit'])
    fig, m_axs = plt.subplots(n_samples, len(grouper), figsize=(20, 4))
    for (c_group, clus_group), c_ims in zip(grouper, m_axs.T):
        c_ims[0].set_title('Group: {}\nSplit: {}'.format(*c_group))
        for (_, clus_row), c_im in zip(clus_group.sample(n_samples, replace=True).iterrows(), c_ims):
            c_im.imshow(clus_row['images'])
            c_im.axis('off')

    fig.savefig('cluster_overview.png')


def visualize_clusters_on_disk(img_df):
    CLUSTER_FOLDER = 'clusters30_agg_avg'

    os.mkdir((ROOT_DIR / 'clustering' / CLUSTER_FOLDER).as_posix())

    for i in range(-1, NUM_CLUSTERS + 1):
        os.mkdir((ROOT_DIR / 'clustering' / CLUSTER_FOLDER / str(i)).as_posix())

    img_df.apply(lambda x:
                 imageio.imsave((ROOT_DIR / 'clustering' / CLUSTER_FOLDER / x['cluster-id'] /
                                 (x['ImageId'])).as_posix(), x['images']), axis=1)

    counts = img_df.groupby(['cluster-id']).size().sort_values(ascending=False)
    print(counts)

    C = list(counts.items())
    C_Sorted = sorted(C, key=lambda x: x[1])


if __name__ == '__main__':
    img_df, roidb = load_images()
    img_df = create_color_features(img_df, roidb)
    img_df = create_color_cluster_agglomerative_clustering(img_df, NUM_CLUSTERS)

    visualize_cluster_samples(img_df)
    visualize_clusters_on_disk(img_df)