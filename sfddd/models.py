import lasagne
from lasagne.layers import InputLayer, Conv2DLayer, MaxPool2DLayer, \
    DenseLayer, dropout


def test_cnn(size_x, size_y, input_var=None):
    net = InputLayer(shape=(None, 1, size_x, size_y),
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
