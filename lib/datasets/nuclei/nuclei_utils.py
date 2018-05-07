from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from scipy import linalg
from PIL import Image


def normalize_and_whiten(I):
    regularization = 1e-5

    if I.shape[2] == 4:
        I = I[:, :, :3]

    # convert to gray by value in HSV
    I = (I.max(-1) + I.min(-1)) / 2.0

    shape = I.shape

    I_scaled = preprocessing.scale(np.asarray(I.flatten(), np.float))
    I_normalized = preprocessing.normalize(I_scaled.reshape(1, -1), norm='l2').reshape(shape)
    X = I_normalized.reshape(-1, 1)

    cov = np.dot(X.T, X) / (X.shape[0] - 1)
    U, S, _ = linalg.svd(cov)
    s = np.sqrt(S.clip(regularization))
    s_inv = np.diag(1. / s)
    whiten_ = np.dot(U, np.dot(s_inv, U.T))
    IZ = np.dot(X, whiten_.T)

    whiten_I = IZ.reshape(shape)
    return np.stack((whiten_I,)*3, -1)


def rescale_0_1(X):
    min_max_scaler = MinMaxScaler(feature_range=(0, 1), copy=False)
    sz = X.shape
    X = min_max_scaler.fit_transform(X.reshape(-1, 1))
    X = X.reshape(sz)
    return X


def rescale_0_255(X):
    min_max_scaler = MinMaxScaler(feature_range=(0, 255), copy=False)
    sz = X.shape
    X = min_max_scaler.fit_transform(X.reshape(-1, 1))
    X = X.reshape(sz)
    return X


def rescale_to_255(i):
    import numpy as np
    m, M = i.min(), i.max()
    I = np.asarray((i - m) / (M - m) * 255, np.uint8)
    return I


def image_ids_in(root_dir):
    ids = []
    for id in os.listdir(root_dir.as_posix()):
        if id == '.DS_Store':
            print('Skipping:', id)
        else:
            ids.append(id)
    return ids


def show_masks(masks, dim=-1):
    M1 = np.sum(masks, axis=dim)
    M1[M1 > 0] = 255
    show(M1)


def show(i):
    i = np.asarray(i, np.float)
    m, M = i.min(), i.max()
    I = np.asarray((i - m) / (M - m + 0.000001) * 255, np.uint8)
    Image.fromarray(I).show()


def save(i, filename):
    i = np.asarray(i, np.float)
    m, M = i.min(), i.max()
    I = np.asarray((i - m) / (M - m + 0.000001) * 255, np.uint8)
    Image.fromarray(I).save(filename)
