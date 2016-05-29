import logging
import pickle

import lasagne
from lasagne.layers import InputLayer, Conv2DLayer, MaxPool2DLayer, \
    DenseLayer, DropoutLayer, dropout, NonlinearityLayer

logger = logging.getLogger(__name__)


def test_cnn(size_x, size_y, input_var=None):
    net = InputLayer(shape=(None, 3, size_x, size_y),
                     input_var=input_var)
    net = Conv2DLayer(net,
                      num_filters=32,
                      filter_size=(5, 5),
                      nonlinearity=lasagne.nonlinearities.rectify,
                      W=lasagne.init.GlorotUniform())
    net = MaxPool2DLayer(net,
                         pool_size=(2, 2))
    net = Conv2DLayer(net,
                      num_filters=32,
                      filter_size=(5, 5),
                      nonlinearity=lasagne.nonlinearities.rectify)
    net = MaxPool2DLayer(net,
                         pool_size=(2, 2))
    net = DenseLayer(dropout(net, p=.5),
                     num_units=256,
                     nonlinearity=lasagne.nonlinearities.rectify)
    net = DenseLayer(dropout(net, p=.5),
                     num_units=10,
                     nonlinearity=lasagne.nonlinearities.softmax)
    return net


def vgg16(input_var=None, w_path='data/vgg16.pkl'):
    net = InputLayer((None, 3, 224, 224), input_var=input_var)
    net = Conv2DLayer(net, 64, 3, pad=1, flip_filters=False)
    net = Conv2DLayer(net, 64, 3, pad=1, flip_filters=False)
    net = MaxPool2DLayer(net, 2)
    net = Conv2DLayer(net, 128, 3, pad=1, flip_filters=False)
    net = Conv2DLayer(net, 128, 3, pad=1, flip_filters=False)
    net = MaxPool2DLayer(net, 2)
    net = Conv2DLayer(net, 256, 3, pad=1, flip_filters=False)
    net = Conv2DLayer(net, 256, 3, pad=1, flip_filters=False)
    net = Conv2DLayer(net, 256, 3, pad=1, flip_filters=False)
    net = MaxPool2DLayer(net, 2)
    net = Conv2DLayer(net, 512, 3, pad=1, flip_filters=False)
    net = Conv2DLayer(net, 512, 3, pad=1, flip_filters=False)
    net = Conv2DLayer(net, 512, 3, pad=1, flip_filters=False)
    net = MaxPool2DLayer(net, 2)
    net = Conv2DLayer(net, 512, 3, pad=1, flip_filters=False)
    net = Conv2DLayer(net, 512, 3, pad=1, flip_filters=False)
    net = Conv2DLayer(net, 512, 3, pad=1, flip_filters=False)
    net = MaxPool2DLayer(net, 2)
    net = DenseLayer(net, num_units=4096)
    net = DropoutLayer(net, p=0.5)
    net = DenseLayer(net, num_units=4096)
    dp_final = DropoutLayer(net, p=0.5)
    fc_final = DenseLayer(dp_final, num_units=1000, nonlinearity=None)
    out_layer = NonlinearityLayer(fc_final, lasagne.nonlinearities.softmax)

    logger.info("Loading vgg16 weights...")
    mdl = pickle.load(open(w_path, 'rb'))
    lasagne.layers.set_all_param_values(out_layer, mdl['param values'])

    net = DenseLayer(dp_final, num_units=10, nonlinearity=None)
    net = NonlinearityLayer(net, lasagne.nonlinearities.softmax)

    return net
