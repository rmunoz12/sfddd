import cPickle
import gzip
import logging
import os
import time

import cv2
import lasagne
import numpy as np
import theano
import theano.tensor as T
from tqdm import tqdm

from sfddd import models
from .preproc import SIZE_X, SIZE_Y
from .util import gpu_free_mem

logger = logging.getLogger(__name__)

DEFAULT_BATCHSIZE = 32
LEARNING_RATE = 0.0001


def rand_rotate(img, low=-20, high=20):
    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, np.random.uniform(low, high), 1)
    img = cv2.warpAffine(img, M, (w, h))
    return img


def rand_translate(img, low=-4, high=4):
    h, w = img.shape[:2]
    shift_x, shift_y = np.random.randint(low, high + 1, size=2)
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    img = cv2.warpAffine(img, M, (w, h))
    return img


def rand_scale(img, low=0.8, high=1.2):
    factor = np.random.uniform(low, high)
    method = cv2.INTER_LINEAR
    if factor < 1:
        method = cv2.INTER_AREA
    img = cv2.resize(img, None, fx=factor, fy=factor, interpolation=method)
    return img


def augment_batch(X):
    for i in range(len(X)):
        img = X[i]
        img = np.transpose(img, (1, 2, 0))
        h, w = img.shape[:2]

        img = rand_scale(img)
        img = rand_translate(img)
        img = rand_rotate(img)

        method = cv2.INTER_LINEAR
        if img.shape[0] < 224:
            method = cv2.INTER_AREA
        img = cv2.resize(img, (w, h), interpolation=method)
        img = np.transpose(img, (2, 0, 1))
        X[i] = img
    return X


def load_img_batch(fnames, cache_folder='cache/train/', augment=False):
    ext = '.pkl.gzip'
    X = []
    for fn in fnames:
        with gzip.open(os.path.join(cache_folder, fn + ext), 'rb') as fi:
            img = cPickle.load(fi)
        X.append(img)
    X = np.array(X).astype('float32')
    X = X.reshape((-1, 3, SIZE_X, SIZE_Y))
    if augment:
        X = augment_batch(X)
    return X


def minibatch_iterator(inputs, targets, batchsize, cache_folder='cache/train/',
                       augment=False):
    assert len(inputs) == len(targets)
    indicies = np.arange(len(inputs))
    np.random.shuffle(indicies)
    for start_idx in tqdm(range(0, len(inputs) - batchsize + 1, batchsize)):
        excerpt = indicies[start_idx:start_idx + batchsize]
        X = load_img_batch(inputs[excerpt], cache_folder, augment=augment)
        yield X, targets[excerpt]


def testbatch_iterator(inputs, batchsize, cache_folder='cache/test/'):
    for start_idx in tqdm(range(0, len(inputs) - batchsize + 1, batchsize)):
        excerpt = slice(start_idx, start_idx + batchsize)
        X = load_img_batch(inputs[excerpt], cache_folder)
        yield X


def train(Xs, Ys, Xv, Yv, size_x=SIZE_X, size_y=SIZE_Y, epochs=2,
          batchsize=DEFAULT_BATCHSIZE, cache_folder='cache/'):
    logger.info('GPU Free Mem: %.3fGB' % gpu_free_mem('gb'))

    cache_folder = os.path.join(cache_folder, 'train/')
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    logger.info("Compiling network functions...")
    # net = models.test_cnn(size_x, size_y, input_var)
    net = models.vgg16(input_var)


    prediction = lasagne.layers.get_output(net)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    params = lasagne.layers.get_all_params(net, trainable=True)
    updates = lasagne.updates.adam(loss, params, learning_rate=LEARNING_RATE)

    test_prediction = lasagne.layers.get_output(net, deterministic=True)
    test_loss = lasagne.objectives.\
                    categorical_crossentropy(test_prediction, target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])
    predict_proba = theano.function([input_var], test_prediction)

    logger.info("Training...")
    logger.info('GPU Free Mem: %.3f' % gpu_free_mem('gb'))

    best_val_loss, best_epoch = None, None

    for epoch in range(epochs):
        start_time = time.time()
        train_err, train_batches = 0, 0
        for batch in minibatch_iterator(Xs, Ys, batchsize, cache_folder, True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        val_err, val_acc, val_batches = 0, 0, 0
        for batch in minibatch_iterator(Xv, Yv, batchsize, cache_folder):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        train_loss = train_err / train_batches
        val_loss = val_err / val_batches
        val_acc = val_acc / val_batches * 100
        end_time = time.time() - start_time

        if not best_val_loss or val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            np.savez('out/model.npz', *lasagne.layers.get_all_param_values(net))

        logger.info("epoch[%d] -- Ls: %.3f | Lv: %.3f | ACCv: %.3f | Ts: %.3f"
                    % (epoch, train_loss, val_loss, val_acc, end_time))

    logger.info("loading best model: epoch[%d]" % best_epoch)
    with np.load('out/model.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(net, param_values)

    return predict_proba


def predict(Xt, pred_fn, batchsize=2, cache_folder='cache/'):
    cache_folder = os.path.join(cache_folder, 'test/')
    logger.info('Predicting on test set...')
    pred = []
    for batch in testbatch_iterator(Xt, batchsize, cache_folder):
        pred.extend(pred_fn(batch))
    pred = np.array(pred)
    logger.info('pred shape: (%d, %d)' % pred.shape)
    return pred
