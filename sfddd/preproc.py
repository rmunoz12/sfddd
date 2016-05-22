import fnmatch
import logging
import os

import cv2
import numpy as np
import pandas as pd
from sklearn.cross_validation import LabelKFold

logger = logging.getLogger(__name__)

# Reshape images from 640 x 480 to
SIZE_X = 128
SIZE_Y = SIZE_X


def get_training_images(folder, drivers, pattern="*.jpg"):
    logger.info("loading training images in: %s" % folder)
    for root, _, fns in os.walk(folder):
        for f in fnmatch.filter(fns, pattern):
            # sub-folders are label names cY, with Y = 0-9
            lbl = os.path.split(root)[1][-1]
            img = cv2.imread(os.path.join(root, f), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (SIZE_X, SIZE_Y))
            driver = drivers[f]
            yield img, lbl, driver


def get_test_images(folder, pattern="*.jpg"):
    logger.info("loading test images in: %s" % folder)
    for root, _, fns in os.walk(folder):
        for f in fnmatch.filter(fns, pattern):
            img = cv2.imread(os.path.join(root, f), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (SIZE_X, SIZE_Y))
            yield img, f


def load_drivers(path):
    df = pd.read_csv(path)
    drivers = {fn: d for fn, d in zip(df['img'], df['subject'])}
    return drivers


def load_train(imgs_folder, drivers_path):
    drivers = load_drivers(drivers_path)

    X, Y, D = [], [], []
    path = os.path.join(imgs_folder, 'train')
    for img, lbl, driver in get_training_images(path, drivers):
        X.append(img)
        Y.append(lbl)
        D.append(driver)
    logger.info("Loaded %d images" % len(Y))

    X = np.array(X).astype('float32') / 255
    X = X.reshape((-1, 1, SIZE_X, SIZE_Y))
    Y = np.array(Y).astype('int32')
    logger.info("X shape: (%d, %d, %d, %d)" % X.shape)
    logger.info("Y shape: (%d,)" % Y.shape)

    logger.info("Splitting local validation set")

    cv = LabelKFold(D, 10)
    index_s, index_t = next(iter(cv))
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
