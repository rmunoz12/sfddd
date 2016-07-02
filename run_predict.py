"""
Predict from snapshot file.

Usage:
    run_predict.py snapshot

Where snapshot was saved by `run_train.py`.  Model type and folders are as
defined in `config.py`.
"""

from argparse import ArgumentParser
import cPickle
import gzip
import logging
import os

import theano.tensor as T

from config import config
from sfddd import models
from sfddd.sgd import Predictor
from sfddd.submit import save_submission
from sfddd.util import timed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def get_args():
    p = ArgumentParser()
    p.add_argument('snapshot', type=str,
                   help='snapshot file containing model weights')
    return p.parse_args()


@timed
def main(args):
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

    predictor = Predictor(batch_size=2)
    pred = predictor.predict(data['Xt'], mdl, snapshot=args.snapshot)
    save_submission(pred, data['test_fnames'], sample_submission,
                    config.paths.out_folder)


if __name__ == '__main__':
    main(get_args())
