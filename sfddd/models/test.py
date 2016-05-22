import lasagne


def test_cnn(size_x, size_y, input_var=None):
    network = lasagne.layers.InputLayer(shape=(None, 1, size_x, size_y),
                                        input_var=input_var)

    network = lasagne.layers.Conv2DLayer(
                  network, num_filters=32, filter_size=(5, 5),
                  nonlinearity=lasagne.nonlinearities.rectify,
                  W=lasagne.init.GlorotUniform())

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = lasagne.layers.Conv2DLayer(
                  network, num_filters=32, filter_size=(5, 5),
                  nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=.5),
        num_units=256,
        nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=.5),
        num_units=10,
        nonlinearity=lasagne.nonlinearities.softmax)

    return network
