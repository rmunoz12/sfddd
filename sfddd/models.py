import logging
import pickle

import cv2
import lasagne
from lasagne.layers import InputLayer, Conv2DLayer, MaxPool2DLayer, \
    DenseLayer, DropoutLayer, NonlinearityLayer, Pool2DLayer, GlobalPoolLayer, \
    ConcatLayer
from lasagne.layers.dnn import Conv2DDNNLayer
from lasagne.layers.normalization import batch_norm
from lasagne.nonlinearities import softmax
from lasagne.utils import floatX
import numpy as np

logger = logging.getLogger(__name__)

# Based on models at
# https://github.com/Lasagne/Recipes/tree/master/modelzoo

VGG_MEAN_VALUE = np.array([103.939, 116.779, 123.68]).reshape((3, 1, 1))  # BGR


class NetModel(object):
    """ Base network model class. """

    def __init__(self, input_var=None, w_path=None):
        """ Build `net` and load weights if `self.w_path` is not None """
        self.input_var = input_var
        self.w_path = w_path
        self.net = None

    def preprocess(self, img):
        """ Convert image pixels into appropriate network input. """
        raise NotImplementedError

    def predict(self, data):
        raise NotImplementedError

    def load_weights(self, top_layer_key):
        """ Load network weights """
        if not self.w_path:
            logger.error("Invalid `w_path`")
            raise ValueError(self.w_path)
        logger.info("Loading network weights: %s" % self.w_path)
        mdl = pickle.load(open(self.w_path, 'rb'))
        lasagne.layers. \
            set_all_param_values(self.net[top_layer_key], mdl['param values'])

    def get_output_layer(self):
        raise NotImplementedError


class Vgg16Base(NetModel):
    """ Vgg16 imagenet classification model """

    def __init__(self, input_var=None, w_path='data/vgg16_rm.pkl'):
        super(Vgg16Base, self).__init__(input_var, w_path)

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
        net['prob'] = NonlinearityLayer(net['fc8'], softmax)

        self.net = net
        if w_path:
            self.load_weights('prob')

    def preprocess(self, img):
        """ img received and returned in BGR -- (c, h, w) """
        img = np.transpose(img, (1, 2, 0))

        img = cv2.resize(img, (256, 256))
        h, w, _ = img.shape
        img = img[h//2-112:h//2+112, w//2-112:w//2+112]

        img = np.transpose(img, (2, 0, 1))
        img = floatX(img)
        img = img - VGG_MEAN_VALUE
        return img

    def get_output_layer(self):
        return self.net['prob']


class Vgg16(Vgg16Base):
    """ Vgg16 model with fc8 replaced with new 10-class classification layer"""

    def __init__(self, input_var=None, w_path='data/vgg16.pkl'):
        super(Vgg16, self).__init__(input_var, w_path)
        net = self.net
        net = DenseLayer(net['drop7'], num_units=10, nonlinearity=None)
        net = NonlinearityLayer(net, lasagne.nonlinearities.softmax)
        self.net = net

    def get_output_layer(self):
        return self.net


class IncV3Base(NetModel):
    """ Google inception V3 model """

    def __init__(self, input_var=None, w_path='data/inception_v3.pkl'):
        super(IncV3Base, self).__init__(input_var, w_path)
        self.net = self.build_network(input_var=input_var)
        if w_path:
            self.load_weights('softmax')

    def preprocess(self, img):
        """
        Expected input: BGR uint8 image
        Converts to RGB, 299x299 pixels, scaled to [-1, 1].
        """
        img = np.transpose(img, (1, 2, 0))

        img = cv2.resize(img, (342, 342))
        h, w, _ = img.shape
        img = img[h//2-150:h//2+149, w//2-150:w//2+149]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = floatX(img)
        img = (img - 128) / 128.0

        img = np.transpose(img, (2, 0, 1))
        return img

    @staticmethod
    def bn_conv(input_layer, **kwargs):
        l = Conv2DLayer(input_layer, **kwargs)
        l = batch_norm(l, epsilon=0.001)
        return l

    @staticmethod
    def inceptionA(input_layer, nfilt):
        # Corresponds to a modified version of figure 5 in the paper
        l1 = IncV3.bn_conv(input_layer, num_filters=nfilt[0][0], filter_size=1)

        l2 = IncV3.bn_conv(input_layer, num_filters=nfilt[1][0], filter_size=1)
        l2 = IncV3.bn_conv(l2, num_filters=nfilt[1][1], filter_size=5, pad=2)

        l3 = IncV3.bn_conv(input_layer, num_filters=nfilt[2][0], filter_size=1)
        l3 = IncV3.bn_conv(l3, num_filters=nfilt[2][1], filter_size=3, pad=1)
        l3 = IncV3.bn_conv(l3, num_filters=nfilt[2][2], filter_size=3, pad=1)

        l4 = Pool2DLayer(
            input_layer, pool_size=3, stride=1, pad=1, mode='average_exc_pad')
        l4 = IncV3.bn_conv(l4, num_filters=nfilt[3][0], filter_size=1)

        return ConcatLayer([l1, l2, l3, l4])

    @staticmethod
    def inceptionB(input_layer, nfilt):
        # Corresponds to a modified version of figure 10 in the paper
        l1 = IncV3.bn_conv(input_layer, num_filters=nfilt[0][0], filter_size=3, stride=2)

        l2 = IncV3.bn_conv(input_layer, num_filters=nfilt[1][0], filter_size=1)
        l2 = IncV3.bn_conv(l2, num_filters=nfilt[1][1], filter_size=3, pad=1)
        l2 = IncV3.bn_conv(l2, num_filters=nfilt[1][2], filter_size=3, stride=2)

        l3 = Pool2DLayer(input_layer, pool_size=3, stride=2)

        return ConcatLayer([l1, l2, l3])

    @staticmethod
    def inceptionC(input_layer, nfilt):
        # Corresponds to figure 6 in the paper
        l1 = IncV3.bn_conv(input_layer, num_filters=nfilt[0][0], filter_size=1)

        l2 = IncV3.bn_conv(input_layer, num_filters=nfilt[1][0], filter_size=1)
        l2 = IncV3.bn_conv(l2, num_filters=nfilt[1][1], filter_size=(1, 7), pad=(0, 3))
        l2 = IncV3.bn_conv(l2, num_filters=nfilt[1][2], filter_size=(7, 1), pad=(3, 0))

        l3 = IncV3.bn_conv(input_layer, num_filters=nfilt[2][0], filter_size=1)
        l3 = IncV3.bn_conv(l3, num_filters=nfilt[2][1], filter_size=(7, 1), pad=(3, 0))
        l3 = IncV3.bn_conv(l3, num_filters=nfilt[2][2], filter_size=(1, 7), pad=(0, 3))
        l3 = IncV3.bn_conv(l3, num_filters=nfilt[2][3], filter_size=(7, 1), pad=(3, 0))
        l3 = IncV3.bn_conv(l3, num_filters=nfilt[2][4], filter_size=(1, 7), pad=(0, 3))

        l4 = Pool2DLayer(
            input_layer, pool_size=3, stride=1, pad=1, mode='average_exc_pad')
        l4 = IncV3.bn_conv(l4, num_filters=nfilt[3][0], filter_size=1)

        return ConcatLayer([l1, l2, l3, l4])

    @staticmethod
    def inceptionD(input_layer, nfilt):
        # Corresponds to a modified version of figure 10 in the paper
        l1 = IncV3.bn_conv(input_layer, num_filters=nfilt[0][0], filter_size=1)
        l1 = IncV3.bn_conv(l1, num_filters=nfilt[0][1], filter_size=3, stride=2)

        l2 = IncV3.bn_conv(input_layer, num_filters=nfilt[1][0], filter_size=1)
        l2 = IncV3.bn_conv(l2, num_filters=nfilt[1][1], filter_size=(1, 7), pad=(0, 3))
        l2 = IncV3.bn_conv(l2, num_filters=nfilt[1][2], filter_size=(7, 1), pad=(3, 0))
        l2 = IncV3.bn_conv(l2, num_filters=nfilt[1][3], filter_size=3, stride=2)

        l3 = Pool2DLayer(input_layer, pool_size=3, stride=2)

        return ConcatLayer([l1, l2, l3])

    @staticmethod
    def inceptionE(input_layer, nfilt, pool_mode):
        # Corresponds to figure 7 in the paper
        l1 = IncV3.bn_conv(input_layer, num_filters=nfilt[0][0], filter_size=1)

        l2 = IncV3.bn_conv(input_layer, num_filters=nfilt[1][0], filter_size=1)
        l2a = IncV3.bn_conv(l2, num_filters=nfilt[1][1], filter_size=(1, 3), pad=(0, 1))
        l2b = IncV3.bn_conv(l2, num_filters=nfilt[1][2], filter_size=(3, 1), pad=(1, 0))

        l3 = IncV3.bn_conv(input_layer, num_filters=nfilt[2][0], filter_size=1)
        l3 = IncV3.bn_conv(l3, num_filters=nfilt[2][1], filter_size=3, pad=1)
        l3a = IncV3.bn_conv(l3, num_filters=nfilt[2][2], filter_size=(1, 3), pad=(0, 1))
        l3b = IncV3.bn_conv(l3, num_filters=nfilt[2][3], filter_size=(3, 1), pad=(1, 0))

        l4 = Pool2DLayer(
            input_layer, pool_size=3, stride=1, pad=1, mode=pool_mode)

        l4 = IncV3.bn_conv(l4, num_filters=nfilt[3][0], filter_size=1)

        return ConcatLayer([l1, l2a, l2b, l3a, l3b, l4])

    @staticmethod
    def build_network(input_var=None):
        net = {}

        net['input'] = InputLayer((None, 3, 299, 299), input_var=input_var)
        net['conv'] = IncV3.bn_conv(net['input'],
                              num_filters=32, filter_size=3, stride=2)
        net['conv_1'] = IncV3.bn_conv(net['conv'], num_filters=32, filter_size=3)
        net['conv_2'] = IncV3.bn_conv(net['conv_1'],
                                num_filters=64, filter_size=3, pad=1)
        net['pool'] = Pool2DLayer(net['conv_2'], pool_size=3, stride=2, mode='max')

        net['conv_3'] = IncV3.bn_conv(net['pool'], num_filters=80, filter_size=1)

        net['conv_4'] = IncV3.bn_conv(net['conv_3'], num_filters=192, filter_size=3)

        net['pool_1'] = Pool2DLayer(net['conv_4'],
                                    pool_size=3, stride=2, mode='max')
        net['mixed/join'] = IncV3.inceptionA(
            net['pool_1'], nfilt=((64,), (48, 64), (64, 96, 96), (32,)))
        net['mixed_1/join'] = IncV3.inceptionA(
            net['mixed/join'], nfilt=((64,), (48, 64), (64, 96, 96), (64,)))

        net['mixed_2/join'] = IncV3.inceptionA(
            net['mixed_1/join'], nfilt=((64,), (48, 64), (64, 96, 96), (64,)))

        net['mixed_3/join'] = IncV3.inceptionB(
            net['mixed_2/join'], nfilt=((384,), (64, 96, 96)))

        net['mixed_4/join'] = IncV3.inceptionC(
            net['mixed_3/join'],
            nfilt=((192,), (128, 128, 192), (128, 128, 128, 128, 192), (192,)))

        net['mixed_5/join'] = IncV3.inceptionC(
            net['mixed_4/join'],
            nfilt=((192,), (160, 160, 192), (160, 160, 160, 160, 192), (192,)))

        net['mixed_6/join'] = IncV3.inceptionC(
            net['mixed_5/join'],
            nfilt=((192,), (160, 160, 192), (160, 160, 160, 160, 192), (192,)))

        net['mixed_7/join'] = IncV3.inceptionC(
            net['mixed_6/join'],
            nfilt=((192,), (192, 192, 192), (192, 192, 192, 192, 192), (192,)))

        net['mixed_8/join'] = IncV3.inceptionD(
            net['mixed_7/join'],
            nfilt=((192, 320), (192, 192, 192, 192)))

        net['mixed_9/join'] = IncV3.inceptionE(
            net['mixed_8/join'],
            nfilt=((320,), (384, 384, 384), (448, 384, 384, 384), (192,)),
            pool_mode='average_exc_pad')

        net['mixed_10/join'] = IncV3.inceptionE(
            net['mixed_9/join'],
            nfilt=((320,), (384, 384, 384), (448, 384, 384, 384), (192,)),
            pool_mode='max')

        net['pool3'] = GlobalPoolLayer(net['mixed_10/join'])

        net['softmax'] = DenseLayer(
            net['pool3'], num_units=1008, nonlinearity=softmax)

        return net

    def get_output_layer(self):
        return self.net['softmax']

class IncV3(IncV3Base):
    """ InvcV3 model with softmax -> new 10-class classification layer"""

    def __init__(self, input_var=None, w_path='data/inception_v3.pkl'):
        super(IncV3, self).__init__(input_var, w_path)
        net = self.net
        net = DenseLayer(net['pool3'], num_units=10, nonlinearity=softmax)
        self.net = net

    def get_output_layer(self):
        return self.net
