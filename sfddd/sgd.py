from collections import OrderedDict
import logging
import os
import time

import lasagne
from lasagne.utils import floatX
import numpy as np
import theano
import theano.tensor as T
from tqdm import tqdm

from .data import FileSystemData
from .util import gpu_free_mem

logger = logging.getLogger(__name__)


class Solver(object):

    def __init__(self, max_iter, batch_size, iter_size, base_lr):
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.iter_size = iter_size
        self.base_lr = base_lr

    def train(self, Xs, Ys, Xv, Yv, net):
        raise NotImplementedError


def init_adam(grads, params):
    """ Modified from lasagne.updates.adam """
    all_grads = grads
    t_prev = theano.shared(floatX(0.))

    m_prev_list = []
    u_prev_list = []
    for param, g_t in zip(params, all_grads):
        value = param.get_value(borrow=True)
        m_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                               broadcastable=param.broadcastable)
        u_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                               broadcastable=param.broadcastable)

        m_prev_list.append(m_prev)
        u_prev_list.append(u_prev)

    return t_prev, m_prev_list, u_prev_list


def step_adam(grads, params, t_prev, m_prev_list, v_prev_list,
              learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """ Modified from lasagne.updates.adam """
    updates = OrderedDict()

    # Using numpy constant to prevent upcasting of float32
    one = floatX(np.array(1))

    t = t_prev.get_value() + 1
    a_t = learning_rate*np.sqrt(one-beta2**t)/(one-beta1**t)

    for param, g_t, m_prev, v_prev in \
            zip(params, grads, m_prev_list, v_prev_list):
        m_t = beta1*m_prev.get_value() + (one-beta1)*g_t
        v_t = beta2*v_prev.get_value() + (one-beta2)*g_t**2
        step = a_t*m_t/(np.sqrt(v_t) + epsilon)

        updates[m_prev] = m_t
        updates[v_prev] = v_t
        updates[param] = param.get_value() - step

    updates[t_prev] = t
    return updates


class SGDSolver(Solver):

    # TODO apply updates only at iter_size
    # TODO track updates:weights ratios

    def __init__(self, max_iter, batch_size, iter_size, base_lr):
        super(SGDSolver, self).__init__(max_iter, batch_size, iter_size,
                                        base_lr)

    def train(self, Xs, Ys, Xv, Yv, mdl,
              data_folder='data/', out_folder='out/'):

        data_folder = os.path.join(data_folder, 'imgs/', 'train/')
        input_var = mdl.input_var
        net = mdl.get_output_layer()
        target_var = T.ivector('targets')

        prediction = lasagne.layers.get_output(net)
        loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
        loss = loss.mean()

        params = lasagne.layers.get_all_params(net, trainable=True)

        grads = T.grad(loss, params)

        test_prediction = lasagne.layers.get_output(net, deterministic=True)
        test_loss = lasagne.objectives. \
            categorical_crossentropy(test_prediction, target_var)
        test_loss = test_loss.mean()
        test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                          dtype=theano.config.floatX)

        logger.info("Compiling network functions...")
        grads_fn = theano.function([input_var, target_var], grads)
        train_fn = theano.function([input_var, target_var], loss)
        val_fn = theano.function([input_var, target_var], [test_loss, test_acc])
        predict_proba = theano.function([input_var], test_prediction)

        logger.info("Training...")
        logger.info('GPU Free Mem: %.3f' % gpu_free_mem('gb'))

        # TODO change to steps
        epochs = self.max_iter / len(Xs)

        best_val_loss, best_epoch = None, None
        best_mdl_path = os.path.join(out_folder, 'best_model.npz')
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

        steps = 0
        for epoch in range(epochs):
            start_time = time.time()
            train_err, train_batches = 0, 0
            data_s = FileSystemData(Xs, Ys, data_folder, self.batch_size,
                                    infinite=False, augment=True, shuffle=True)
            step_err, step_g = 0, None

            for batch in tqdm(data_s, total=data_s.steps, leave=False):
                inputs, targets = batch
                inputs = floatX(np.array([mdl.preprocess(x) for x in inputs]))

                batch_err = train_fn(inputs, targets)
                batch_g = grads_fn(inputs, targets)

                if step_g is None:
                    step_g = batch_g
                else:
                    step_g = [s_g + b_g for s_g, b_g in zip(step_g, batch_g)]
                train_err += batch_err
                step_err += batch_err
                train_batches += 1
                if train_batches % self.iter_size == 0:
                    step_g = [g / np.array(self.iter_size) for g in step_g]

                    if steps == 0:
                        t_prev, m_prev, u_prev = \
                            init_adam(batch_g, params)
                    updates = step_adam(step_g, params, t_prev, m_prev, u_prev,
                                        learning_rate=self.base_lr)
                    for p, new_val in updates.items():
                        p.set_value(new_val)
                    steps += 1
                    step_err, step_g = 0, None

            data_v = FileSystemData(Xv, Yv, data_folder, self.batch_size,
                                    infinite=False, augment=False, shuffle=False)
            val_err, val_acc, val_batches = 0, 0, 0
            for batch in tqdm(data_v, total=data_v.steps, leave=False):
                inputs, targets = batch
                inputs = floatX(np.array([mdl.preprocess(x) for x in inputs]))
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
                np.savez(best_mdl_path,
                         *lasagne.layers.get_all_param_values(net))
            snapshot_path = os.path.join(out_folder, 'snapshot_epoch_%d.npz'
                                         % epoch)
            np.savez(snapshot_path, *lasagne.layers.get_all_param_values(net))

            logger.info("epoch[%d] -- Ls: %.3f | Lv: %.3f | ACCv: %.3f | Ts: %.3f"
                        % (epoch, train_loss, val_loss, val_acc, end_time))

        logger.info("loading best model: epoch[%d]" % best_epoch)
        with np.load(best_mdl_path) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(net, param_values)

        return predict_proba

    def predict(self, Xt, pred_fn, mdl, batchsize=2, data_folder='data/'):
        data_folder = os.path.join(data_folder, 'imgs/', 'test/')
        logger.info('Predicting on test set...')
        pred = []
        data_t = FileSystemData(Xt, None, data_folder, batch_size=batchsize)
        for batch in tqdm(data_t, total=data_t.steps, leave=False):
            inputs, _ = batch
            inputs = floatX(np.array([mdl.preprocess(x) for x in inputs]))
            pred.extend(pred_fn(inputs))
        pred = np.array(pred)
        logger.info('pred shape: (%d, %d)' % pred.shape)
        return pred


class Predictor(Solver):

    def __init__(self, batch_size=2):
        super(Predictor, self).__init__(None, None, None, None)
        self.batch_size = batch_size

    @staticmethod
    def _compile(mdl):
        input_var = mdl.input_var
        net = mdl.get_output_layer()

        test_prediction = lasagne.layers.get_output(net, deterministic=True)

        logger.info("Compiling network functions...")
        predict_proba = theano.function([input_var], test_prediction)
        return predict_proba

    @staticmethod
    def _load_snapshot(mdl, fp):
        logger.info("loading snapshot: %s" % fp)
        with np.load(fp) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        net = mdl.get_output_layer()
        lasagne.layers.set_all_param_values(net, param_values)


    def predict(self, Xt, mdl, snapshot=None, data_folder='data/'):
        data_folder = os.path.join(data_folder, 'imgs/', 'test/')

        pred_fn = self._compile(mdl)
        if snapshot:
            self._load_snapshot(mdl, snapshot)

        logger.info('Predicting on test set...')
        pred = []
        data_t = FileSystemData(Xt, None, data_folder,
                                batch_size=self.batch_size)
        for batch in tqdm(data_t, total=data_t.steps, leave=False):
            inputs, _ = batch
            inputs = floatX(np.array([mdl.preprocess(x) for x in inputs]))
            pred.extend(pred_fn(inputs))
        pred = np.array(pred)
        logger.info('pred shape: (%d, %d)' % pred.shape)
        return pred
