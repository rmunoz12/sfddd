import logging
import time

import lasagne
import numpy as np
import theano
import theano.tensor as T

from .models.test import test_cnn
from .preproc import SIZE_X, SIZE_Y

logger = logging.getLogger(__name__)

DEFAULT_BATCHSIZE = 32
LEARNING_RATE = 0.001


def minibatch_iterator(inputs, targets, batchsize):
    """Based on Lasagne documentation"""
    assert len(inputs) == len(targets)
    indicies = np.arange(len(inputs))
    np.random.shuffle(indicies)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        excerpt = indicies[start_idx:start_idx + batchsize]
        yield inputs[excerpt], targets[excerpt]


def testbatch_iterator(inputs, batchsize):
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt]


def train(Xs, Ys, Xv, Yv, size_x=SIZE_X, size_y=SIZE_Y, epochs=10,
          batchsize=DEFAULT_BATCHSIZE):
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    logger.info("Compiling network functions...")
    network = test_cnn(size_x, size_y, input_var)
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.adam(loss, params, learning_rate=LEARNING_RATE)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.\
                    categorical_crossentropy(test_prediction, target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])
    predict_proba = theano.function([input_var], test_prediction)

    for epoch in range(epochs):
        start_time = time.time()
        train_err, train_batches = 0, 0
        for batch in minibatch_iterator(Xs, Ys, batchsize):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        val_err, val_acc, val_batches = 0, 0, 0
        for batch in minibatch_iterator(Xv, Yv, batchsize):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        train_loss = train_err / train_batches
        val_loss = val_err / val_batches
        val_acc = val_acc / val_batches * 100
        end_time = time.time() - start_time

        logger.info("epoch[%d] -- Ls: %.3f | Lv: %.3f | ACCv: %.3f | Ts: %.3f"
                    % (epoch, train_loss, val_loss, val_acc, end_time))

    return predict_proba


def predict(Xt, pred_fn, batchsize=2):
    logger.info('Predicting on test set...')
    pred = []
    for batch in testbatch_iterator(Xt, batchsize):
        pred.extend(pred_fn(batch))
    pred = np.array(pred)
    logger.info('pred shape: (%d, %d)' % pred.shape)
    return pred
