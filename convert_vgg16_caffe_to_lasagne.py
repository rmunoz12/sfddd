"""
Convert original VGG-16 model from caffe to lasagne.

Caffe model downloaded from: http://www.robots.ox.ac.uk/~vgg/research/very_deep/
    - VGG_ILSVRC_16_layers.caffemodel
    - VGG_ILSVRC_16_layers_deploy.prototxt


Reference:
https://github.com/Lasagne/Recipes/blob/master/examples/Using%20a%20Caffe%20Pretrained%20Network%20-%20CIFAR10.ipynb
"""

from __future__ import print_function
from argparse import ArgumentParser
import cPickle
import gzip
import logging
from StringIO import StringIO

import caffe
import cv2
import lasagne
from lasagne.layers import DenseLayer, DropoutLayer, InputLayer, \
    MaxPool2DLayer, NonlinearityLayer
from lasagne.layers.dnn import Conv2DDNNLayer
from lasagne.utils import floatX
import numpy as np
from PIL import Image
import requests
import theano
import theano.tensor as T

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

MEAN_VALUE = np.array([103.939, 116.779, 123.68]).reshape((3, 1, 1))  # BGR

LABELS = [267, 140, 24, 559, 833, 829, 924, 839, 512, 388]
SYNSETS = ['n02113799', 'n02027492', 'n01622779', 'n03376595', 'n04347754',
           'n04335435', 'n07583066', 'n04366367', 'n03109150', 'n02510455']


def get_args():
    p = ArgumentParser(description="Convert VGG-16 from caffe to lasagne")
    p.add_argument("prototxt", help="vgg16 caffemodel file")
    p.add_argument("caffemodel", help="vgg16 caffemodel file")
    p.add_argument("output", help="file to save lasagne weights")
    args = p.parse_args()
    return args


def get_caffe_net(prototxt_path, caffemodel_path):
    net = caffe.Net(prototxt_path, caffemodel_path, caffe.TEST)
    return net


def get_lasagne_net(input_var=None):
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
    return net


def convert(net, layers_caffe):
    for name, layer in net.items():
        try:
            caffe_layer = layers_caffe[name]
            if caffe_layer.type in {'Dropout', 'Input', 'Pooling', 'Softmax'}:
                continue

            blob_w = caffe_layer.blobs[0]
            blob_b = caffe_layer.blobs[1]
            w = np.array(blob_w.data)
            b = np.array(blob_b.data)

            if name in {'fc6', 'fc7', 'fc8'}:
                w = w.T  # fix shape errors when running model

            layer.W.set_value(w)
            layer.b.set_value(b)

            logger.debug("Transferred: %s" % name)
        except AttributeError:
            continue


def download_images(synsets=SYNSETS):
    imgs = []
    base_url = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid='
    for s in synsets:
        url = base_url + s
        rq = requests.get(url)
        if rq.status_code != 200:
            logger.debug("Request status: %d" % rq.status_code)
            raise ValueError
        links = [x.strip() for x in rq.text.split('\n')]
        for i in range(len(links)):
            url = links[i]
            rq = requests.get(url, stream=True)
            if rq.status_code != 200:
                if  i == len(links) - 1:
                    raise ValueError
                else:
                    continue
            try:
                img = Image.open(StringIO(rq.content))
                img = np.array(img)
                img = img[:, :, ::-1]
                img = cv2.resize(img, (224, 224))
                img = img.transpose((2, 0, 1))
                img = img - MEAN_VALUE
                imgs.append(img)
                break
            except IndexError:
                logger.debug("Error reading image")
                if i == len(links) - 1:
                    raise ValueError
                else:
                    continue
    imgs = floatX(imgs)
    return imgs


def check_output(nl, nc, imgs, lbls):
    pl = np.array(lasagne.layers.get_output(nl['prob'], imgs, deterministic=True).eval())
    predl = np.argmax(pl, axis=1)

    # nc.blobs['data'].reshape(10, 3, 224, 224)
    nc.blobs['data'].data[:] = imgs
    pc = nc.forward()['prob']
    predc = np.argmax(pc, axis=1)

    result = np.allclose(pl, pc)
    if result:
        logger.info("Caffe and Lasage models probabilities.. OK")
    else:
        logger.warn("Caffe and Lasage models probabilities.. NOT CLOSE")

    result = np.allclose(predl, predc)
    if result:
        logger.info("Caffe and Lasage models predictions.. OK")
    else:
        logger.warn("Caffe and Lasage models predictions.. NOT CLOSE")

    acc = np.mean(predl == lbls)
    logger.info("Lasage model accuracy: %.3f" % acc)


def main():
    args = get_args()

    logger.info('loading caffe model...')
    net_caffe = \
        get_caffe_net(args.prototxt, args.caffemodel)
    layers_caffe = dict(zip(list(net_caffe._layer_names), net_caffe.layers))

    logger.info('compiling lasagne model...')
    input_var = T.tensor4('inputs')
    net = get_lasagne_net(input_var)
    test_prediction = lasagne.layers.get_output(net['prob'], deterministic=True)

    logger.info('converting caffe --> lasagne...')
    convert(net, layers_caffe)

    logger.info('downloading test images...')
    imgs = download_images()

    logger.info('validating converted model...')
    check_output(net, net_caffe, imgs, LABELS)

    logger.info("Saving model weights...")
    values = lasagne.layers.get_all_param_values(net['prob'])
    cPickle.dump(values, gzip.open(args.output, 'wb'))


if __name__ == '__main__':
    main()
