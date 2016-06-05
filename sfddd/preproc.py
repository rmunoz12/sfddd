from __future__ import division
import cPickle
import fnmatch
import gzip
import logging
import os

import cv2
from lasagne.utils import floatX
import matplotlib.image
import numpy as np
import pandas as pd
import skimage.transform
from sklearn.cross_validation import LabelKFold
from sklearn.utils import shuffle
from tqdm import tqdm

logger = logging.getLogger(__name__)

MEAN_VALUE = np.array([103.939, 116.779, 123.68]).reshape((3, 1, 1))  # BGR

# Reshape images from 640 x 480 to
SIZE_X = 224
SIZE_Y = SIZE_X


# based on https://github.com/Lasagne/Recipes/blob/master/examples/ImageNet%20Pretrained%20Network%20(VGG_S).ipynb
def preproc_img(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (480, 480), interpolation=cv2.INTER_AREA)
    img = np.transpose(img, (2, 0, 1))

    # img = matplotlib.image.imread(path)
    # h, w, _ = img.shape
    # img = skimage.transform.resize(img, (480, w * 480 // h),
    #                                preserve_range=True)
    #


    # Central crop to 224x224
    # h, w, _ = img.shape
    # img = img[h//2-112:h//2+112, w//2-112:w//2+112]

    # rawimg = np.copy(img).astype('uint8')

    # Shuffle axes to c01
    # img = np.swapaxes(np.swapaxes(img, 1, 2), 0, 1)

    # Convert to BGR
    # img = img[::-1, :, :]

    img = img - MEAN_VALUE
    return floatX(img[np.newaxis])


def get_training_images(folder, drivers, pattern="*.jpg"):
    for root, _, fns in os.walk(folder):
        logger.info("loading training images in: %s" % root)
        matches = fnmatch.filter(fns, pattern)
        for f in tqdm(matches):
            # sub-folders are label names cY, with Y = 0-9
            lbl = os.path.split(root)[1][-1]
            img = preproc_img(os.path.join(root, f))
            driver = drivers[f]
            yield img, lbl, driver, f


def get_test_images(folder, pattern="*.jpg"):
    for root, _, fns in os.walk(folder):
        logger.info("loading test images in: %s" % root)
        matches = fnmatch.filter(fns, pattern)
        for f in tqdm(matches):
            img = preproc_img(os.path.join(root, f))
            yield img, f


def load_drivers(path):
    df = pd.read_csv(path)
    drivers = {fn: d for fn, d in zip(df['img'], df['subject'])}
    return drivers


def load_train(imgs_folder, drivers_path, cache_folder="cache/"):
    drivers = load_drivers(drivers_path)

    cache_folder = os.path.join(cache_folder, 'train')
    if not os.path.exists(cache_folder) and cache_folder != '':
        os.makedirs(cache_folder)

    X, Y, D = [], [], []
    path = os.path.join(imgs_folder, 'train')
    for img, lbl, driver, fn in get_training_images(path, drivers):
        fo = gzip.open(os.path.join(cache_folder, fn + '.pkl.gzip'), 'wb')
        cPickle.dump(img, fo)
        fo.close()
        X.append(fn)
        Y.append(lbl)
        D.append(driver)
    logger.info("Loaded %d images" % len(Y))

    # X = np.array(X).astype('float32')
    # X = X.reshape((-1, 3, SIZE_X, SIZE_Y))
    X = np.array(X)
    Y = np.array(Y).astype('int32')
    # logger.info("X shape: (%d, %d, %d, %d)" % X.shape)
    logger.info("X shape: (%d,)" % X.shape)
    logger.info("Y shape: (%d,)" % Y.shape)

    logger.info("Splitting local validation set")
    X, Y, D = shuffle(X, Y, D)
    cv = LabelKFold(D, 10)
    index_s, index_t = next(iter(cv))
    Xs, Xt = X[index_s], X[index_t]
    Ys, Yt = Y[index_s], Y[index_t]

    return Xs, Ys, Xt, Yt


def load_test(imgs_folder, cache_folder='cache/'):
    cache_folder = os.path.join(cache_folder, 'test')
    if not os.path.exists(cache_folder) and cache_folder != '':
        os.makedirs(cache_folder)

    X = []
    fnames = []
    path = os.path.join(imgs_folder, 'test')
    for img, fn in get_test_images(path):
        fo = gzip.open(os.path.join(cache_folder, fn + '.pkl.gzip'), 'wb')
        cPickle.dump(img, fo)
        fo.close()
        X.append(fn)
        fnames.append(fn)
    logger.info("Loaded %d images" % len(X))

    # X = np.array(X).astype('float32')
    # X = X.reshape((-1, 3, SIZE_X, SIZE_Y))
    X = np.array(X)
    fnames = np.array(fnames)
    # logger.info("Xt shape: (%d, %d, %d, %d)" % X.shape)
    logger.info("Xt shape: (%d,)" % X.shape)
    logger.info("fnames shape: (%d,)" % fnames.shape)

    return X, fnames
