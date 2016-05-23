import cPickle
import gzip
import logging
import os

from config import config
from sfddd.preproc import load_train, load_test
from sfddd.util import timed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


@timed
def main():
    imgs_folder = os.path.join(config.paths.data_folder, 'imgs')
    drivers_path = os.path.join(config.paths.data_folder,
                                'driver_imgs_list.csv')

    Xs, Ys, Xv, Yv = load_train(imgs_folder, drivers_path)
    names = ['Xs', 'Ys', 'Xv', 'Yv']
    for name, obj in zip(names, [Xs, Ys, Xv, Yv]):
        fn = name + '.pkl'
        with gzip.open(os.path.join(config.paths.cache_folder, fn)) as fo:
            cPickle.dump(obj, fo)
    logger.info('Cached training and validation data')

    Xt, test_fnames = load_test(imgs_folder)
    names = ['Xt', 'test_fnames']
    for name, obj in zip(names, [Xt, test_fnames]):
        fn = name + '.pkl'
        with gzip.open(os.path.join(config.paths.cache_folder, fn), 'wb') as fo:
            cPickle.dump(obj, fo)
    logger.info('Cached test data')


if __name__ == '__main__':
    main()
