import logging
import pickle

import lasagne
from lasagne.layers import InputLayer, Conv2DLayer, MaxPool2DLayer, \
    DenseLayer, DropoutLayer, dropout, NonlinearityLayer
from lasagne.layers.dnn import Conv2DDNNLayer

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


def vgg16_base(input_var=None, w_path='data/vgg16_rm.pkl'):
    net = {}
    net['input'] = InputLayer((None, 3, 224, 224), input_var=input_var)
    net['conv1_1'] = Conv2DDNNLayer(net['input'], 64, 3,
                                    pad=1, flip_filters=False)
    net['conv1_2'] = Conv2DDNNLayer(net['conv1_1'], 64, 3,
                                    pad=1, flip_filters=False)
    net['pool1'] = MaxPool2DLayer(net['conv1_2'], 2)
    net['conv2_1'] = Conv2DDNNLayer(net['pool1'], 128, 3,
                                    pad=1, flip_filters=False)
    net['conv2_2'] = Conv2DDNNLayer(net['conv2_1'], 128, 3,
                                    pad=1, flip_filters=False)
    net['pool2'] = MaxPool2DLayer(net['conv2_2'], 2)
    net['conv3_1'] = Conv2DDNNLayer(net['pool2'], 256, 3,
                                    pad=1, flip_filters=False)
    net['conv3_2'] = Conv2DDNNLayer(net['conv3_1'], 256, 3,
                                    pad=1, flip_filters=False)
    net['conv3_3'] = Conv2DDNNLayer(net['conv3_2'], 256, 3,
                                    pad=1, flip_filters=False)
    net['pool3'] = MaxPool2DLayer(net['conv3_3'], 2)
    net['conv4_1'] = Conv2DDNNLayer(net['pool3'], 512, 3,
                                    pad=1, flip_filters=False)
    net['conv4_2'] = Conv2DDNNLayer(net['conv4_1'], 512, 3,
                                    pad=1, flip_filters=False)
    net['conv4_3'] = Conv2DDNNLayer(net['conv4_2'], 512, 3,
                                    pad=1, flip_filters=False)
    net['pool4'] = MaxPool2DLayer(net['conv4_3'], 2)
    net['conv5_1'] = Conv2DDNNLayer(net['pool4'], 512, 3,
                                    pad=1, flip_filters=False)
    net['conv5_2'] = Conv2DDNNLayer(net['conv5_1'], 512, 3,
                                    pad=1, flip_filters=False)
    net['conv5_3'] = Conv2DDNNLayer(net['conv5_2'], 512, 3,
                                    pad=1, flip_filters=False)
    net['pool5'] = MaxPool2DLayer(net['conv5_3'], 2)
    net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
    net['drop6'] = DropoutLayer(net['fc6'], p=0.5)
    net['fc7'] = DenseLayer(net['drop6'], num_units=4096)
    net['drop7'] = DropoutLayer(net['fc7'], p=0.5)
    net['fc8'] = DenseLayer(net['drop7'], num_units=1000,
                            nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc8'], lasagne.nonlinearities.softmax)

    logger.info("Loading vgg16 weights...")
    if w_path:
        mdl = pickle.load(open(w_path, 'rb'))
        lasagne.layers.set_all_param_values(net['prob'], mdl['param values'])

    return net


def vgg16(input_var=None, w_path='data/vgg16.pkl'):
    net = vgg16_base(input_var, w_path)
    net = DenseLayer(net['drop7'], num_units=10, nonlinearity=None)
    net = NonlinearityLayer(net, lasagne.nonlinearities.softmax)
    return net
