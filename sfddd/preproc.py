import fnmatch
import logging
import os

import cv2
import numpy as np
from sklearn.cross_validation import StratifiedShuffleSplit

logger = logging.getLogger(__name__)

# Reshape images from 640 x 480 to
SIZE_X = 128
SIZE_Y = SIZE_X


def get_training_images(folder, pattern="*.jpg"):
    logger.info("loading training images in: %s" % folder)
    for root, _, fns in os.walk(folder):
        for f in fnmatch.filter(fns, pattern):
            # sub-folders are label names cY, with Y = 0-9
            lbl = os.path.split(root)[1][-1]
            img = cv2.imread(os.path.join(root, f), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (SIZE_X, SIZE_Y))
            yield img, lbl


def get_test_images(folder, pattern="*.jpg"):
    logger.info("loading test images in: %s" % folder)
    for root, _, fns in os.walk(folder):
        for f in fnmatch.filter(fns, pattern):
            img = cv2.imread(os.path.join(root, f), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (SIZE_X, SIZE_Y))
            yield img, f


def load_train(imgs_folder, cache_folder=None, test_size=0.1, seed=700):
    X, Y = [], []
    path = os.path.join(imgs_folder, 'train')
    for img, lbl in get_training_images(path):
        X.append(img)
        Y.append(lbl)
    logger.info("Loaded %d images" % len(Y))

    X = np.array(X).astype('float32') / 255
    X = X.reshape((-1, 1, SIZE_X, SIZE_Y))
    Y = np.array(Y).astype('int32')
    logger.info("X shape: (%d, %d, %d, %d)" % X.shape)
    logger.info("Y shape: (%d,)" % Y.shape)

    logger.info("Splitting local validation set: %.2f%%" % test_size)
    sss = StratifiedShuffleSplit(Y, n_iter=1, test_size=test_size,
                                 random_state=seed)
    for index_s, index_t in sss:
        Xs, Xt = X[index_s], X[index_t]
        Ys, Yt = Y[index_s], Y[index_t]

    return Xs, Ys, Xt, Yt


def load_test(imgs_folder):
    X = []
    fnames = []
    path = os.path.join(imgs_folder, 'test')
    for img, fname in get_test_images(path):
        X.append(img)
        fnames.append(fname)
    logger.info("Loaded %d images" % len(X))

    X = np.array(X).astype('float32') / 255
    X = X.reshape((-1, 1, SIZE_X, SIZE_Y))
    fnames = np.array(fnames)
    logger.info("Xt shape: (%d, %d, %d, %d)" % X.shape)
    logger.info("fnames shape: (%d,)" % fnames.shape)

    return X, fnames
