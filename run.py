import logging
import os
from time import time

from config import config
from sfddd.preproc import load_train, load_test
from sfddd.sgd import train, predict
from sfddd.submit import save_submission

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def main():
    t0 = time()

    imgs_folder = os.path.join(config.paths.data_folder, 'imgs')
    sample_submission = os.path.join(config.paths.data_folder,
                                     'sample_submission.csv')
    drivers_path = os.path.join(config.paths.data_folder,
                                'driver_imgs_list.csv')
    Xs, Ys, Xv, Yv = load_train(imgs_folder, drivers_path)
    pred_fn = train(Xs, Ys, Xv, Yv)

    Xt, test_fnames = load_test(imgs_folder)
    pred = predict(Xt, pred_fn)
    save_submission(pred, test_fnames, sample_submission,
                    config.paths.out_folder)

    logger.info('--- %.3f minutes ---' % ((time() - t0) / 60))


if __name__ == '__main__':
    main()
