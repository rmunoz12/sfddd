import cPickle
import gzip
import logging
import os

import theano.tensor as T

from config import config
from sfddd import models
from sfddd.sgd import SGDSolver
from sfddd.submit import save_submission
from sfddd.util import timed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


@timed
def main():
    sample_submission = os.path.join(config.paths.data_folder,
                                     'sample_submission.csv')

    names = ['Xs', 'Ys', 'Xv', 'Yv', 'Xt', 'test_fnames']
    data = {}
    for name in names:
        fn = name + '.pkl.gz'
        with gzip.open(os.path.join(config.paths.cache_folder, fn)) as fi:
            data[name] = cPickle.load(fi)

    input_var = T.tensor4('inputs')

    if config.model == 'vgg':
        mdl = models.Vgg16(input_var)
    elif config.model == 'inc':
        mdl = models.IncV3(input_var)
    else:
        logger.error("Unrecognized model name: %s" % config.model)
        raise ValueError(config.model)

    solver = SGDSolver(max_iter=len(data['Xs']) * 6,
                       batch_size=24, iter_size=24, base_lr=0.0001)

    pred_fn = solver.train(data['Xs'], data['Ys'], data['Xv'], data['Yv'], mdl)
    pred = solver.predict(data['Xt'], pred_fn, mdl)

    save_submission(pred, data['test_fnames'], sample_submission,
                    config.paths.out_folder)


if __name__ == '__main__':
    main()
