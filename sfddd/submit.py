import csv
import logging
import os
import time

logger = logging.getLogger(__name__)


def save_submission(pred, test_fnames, sample_submission, folder):
    if not os.path.exists(folder) and folder != '':
        os.makedirs(folder)

    with open(sample_submission, 'rb') as hfile:
        rdr = csv.reader(hfile)
        header = rdr.next()
    logger.info("header: %s" % header)

    timestamp = time.strftime("%Y-%m-%d-%H%M", time.localtime())
    out_path = os.path.join(folder, 'submission-%s.csv' % timestamp)
    with open(out_path, 'wb') as fo:
        fo.write(','.join(header) + '\n')
        for fn, p in zip(test_fnames, pred):
            fo.write(fn + ',')
            fo.write(','.join(p.astype('str')) + '\n')
    logger.info('saved submission file: %s' % out_path)
