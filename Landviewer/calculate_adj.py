import os, csv, re
import pandas as pd
import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA
import numba
import cv2
from skimage import io, img_as_float32, morphology, exposure
from skimage.feature import graycomatrix, graycoprops
from sklearn.metrics import pairwise_distances
import scipy.sparse as sp


@numba.njit("f4(f4[:], f4[:])")
def euclid_dist(t1, t2):
    sum = 0
    for i in range(t1.shape[0]):
        sum += (t1[i] - t2[i]) ** 2
    return np.sqrt(sum)


@numba.njit("f4[:,:](f4[:,:])", parallel=True, nogil=True)
def pairwise_distance(X):
    n = X.shape[0]
    adj = np.empty((n, n), dtype=np.float32)
    for i in numba.prange(n):
        for j in numba.prange(n):
            adj[i][j] = euclid_dist(X[i], X[j])
    return adj


def calculate_location_adj(x, y, l):
    # x,y,x_pixel, y_pixel are lists
    print("Calculateing location adj matrix")
    X = np.array([x, y]).T.astype(np.float32)
    adj = pairwise_distance(X)
    adj_exp = np.exp(-1 * (adj ** 2) / (2 * (l ** 2)))
    return adj_exp


def extract_image_feature(img_tif, x, y, r=256, dotsize=0):
    img = io.imread(img_tif)
    if dotsize != 0:
        img_new = img.copy()
        for i in range(len(x)):
            x = x[i]
            y = y[i]
            img_new[int(x - dotsize):int(x + dotsize), int(y - dotsize):int(y + dotsize), :] = 255

        cv2.imwrite("stout/map.jpg", img_new)

    img = img_as_float32(img)
    img = (20 * img).astype("uint8")

    feature_set = [
        "contrast",
        "dissimilarity",
        "homogeneity",
        "ASM",
        "energy",
        "correlation",
    ]
    features = []
    for i in range(len(x)):
        if (i + 1) % 100 == 0:
            print("Extract image: processing {} spot out of {} spots".format(i + 1, len(x)))
        spot_img = img[x[i] - r: x[i] + r + 1, y[i] - r: y[i] + r + 1]
        spot_mask = morphology.disk(r)
        # only use the spot, not the bbox
        spot_img = np.einsum("ij,ijk->ijk", spot_mask, spot_img)

        # extract texture features
        rgbfeature = []
        for c in range(img.shape[2]):
            glcm = graycomatrix(
                spot_img[:, :, c],
                distances=[1],
                # Angles are arranged in a counter clockwise manner, in radian.
                angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                levels=21,
                symmetric=True,
                normed=False,
            )
            glcm = glcm[1:, 1:]
            glcm = glcm / np.sum(glcm, axis=(0, 1))
            glcm = np.ravel(np.mean(glcm, axis=(3)))
            rgbfeature.append(glcm)

        spotfeature = np.concatenate(rgbfeature, axis=0)

        features.append(spotfeature)

    return np.concatenate(features, axis=0).reshape((-1, 1200))


def features_construct_graph(features, pca=50, l=100, metric="cosine"):
    from scipy.sparse import issparse
    if issparse(features):
        features = features.toarray()
    if pca is not None:
        features = PCA(n_components=pca).fit_transform(features.toarray())
    # print("k: ", k)
    # print("features_construct_graph features", features.shape)
    fadj = pairwise_distances(features, metric=metric)
    fadj = np.exp(-1 * (fadj ** 2) / (2 * (l ** 2)))

    # nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)
    return fadj  # , nfadj
