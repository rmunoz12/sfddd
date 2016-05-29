import csv
import gzip
import logging
import os
import time

logger = logging.getLogger(__name__)


def save_submission(pred, test_fnames, sample_submission, folder,
                    loss=None, acc=None):
    if not os.path.exists(folder) and folder != '':
        os.makedirs(folder)

    with open(sample_submission, 'rb') as hfile:
        rdr = csv.reader(hfile)
        header = rdr.next()

    timestamp = time.strftime("%Y-%m-%d-%H%M", time.localtime())
    fn = 'submission-%s' % timestamp
    if acc:
        fn += '-%.3f' % acc
    if loss:
        fn += '-%.3f' % loss
    fn += 'csv.gz'
    out_path = os.path.join(folder, fn)
    with gzip.open(out_path, 'wb') as fo:
        fo.write(','.join(header) + '\n')
        for fn, p in zip(test_fnames, pred):
            fo.write(fn + ',')
            fo.write(','.join(p.astype('str')) + '\n')
    logger.info('saved submission file: %s' % out_path)
