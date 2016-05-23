import cPickle
import gzip
import logging
import os

from config import config
from sfddd.sgd import train, predict
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
        fn = name + '.pkl'
        with gzip.open(os.path.join(config.paths.cache_folder, fn)) as fi:
            data[name] = cPickle.load(fi)
    pred_fn = train(data['Xs'], data['Ys'], data['Xv'], data['Yv'])

    pred = predict(data['Xt'], pred_fn)
    save_submission(pred, data['test_fnames'], sample_submission,
                    config.paths.out_folder)


if __name__ == '__main__':
    main()
