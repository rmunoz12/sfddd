import logging
import math
from multiprocessing import Pool

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class DataSet(object):
    """ Base class for iterating over a dataset. """

    def __init__(self, x, y, batch_size=1, infinite=False, shuffle=False):
        self.x = x
        self.y = y
        self.infinite = infinite

        self.n = len(x)
        if y is not None:
            assert len(x) == len(y)

        assert batch_size > 0
        assert batch_size <= self.n
        self.batch_size = batch_size

        self.current = 0
        self.shuffle = shuffle
        if shuffle:
            self.shuffle_data()

        self.steps = None
        if not infinite:
            self.steps = self.n / batch_size

    def load_batch(self):
        """
        Return batch pointed to by `self.current` and raise `StopIteration` if
        loading a batch is not possible.
        """
        raise NotImplementedError

    def shuffle_data(self):
        """ Shuffle data held in `self.x` and `self.y` or slicing method. """
        raise NotImplementedError

    def reset(self):
        if self.shuffle:
            self.shuffle_data()
        self.current = 0

    def __iter__(self):
        return self

    def next(self):
        if not self.infinite and self.current > self.n:
            raise StopIteration
        else:
            try:
                batch = self.load_batch()
            except StopIteration:
                if not self.infinite:
                    raise StopIteration
                else:
                    self.reset()
                    return self.next()
            return batch


class FileSystemData(DataSet):

    def __init__(self, x, y, data_folder,
                 batch_size=1, infinite=False, augment=False, shuffle=False):
        self.indicies = np.arange(len(x))
        self.start_idx = 0
        self.data_folder = data_folder
        self.augment = augment
        super(FileSystemData, self).__init__(x, y,
                                             batch_size, infinite, shuffle)

    def load_batch(self):
        if self.start_idx > self.n - self.batch_size:
            raise StopIteration
        excerpt = self.indicies[self.start_idx:self.start_idx + self.batch_size]

        fnames = self.x[excerpt]
        X = []
        for fn in fnames:
            path = fn
            img = cv2.imread(path, cv2.IMREAD_COLOR)  # shape: (h, w, c)
            img = np.transpose(img, (2, 0, 1))        # shape: (c, h, w)
            X.append(img)

        X = np.array(X)
        if self.augment:
            X = augment_batch(X, resize_only=False)
        else:
            X = augment_batch(X, resize_only=True)

        self.current += self.batch_size

        if self.y is None:
            Y = None
        else:
            Y = self.y[excerpt]

        return X, Y

    def shuffle_data(self):
        np.random.shuffle(self.indicies)


def rand_rotate(img, low=-10, high=10):
    """ Rotate and crop out """
    alpha = np.random.uniform(low, high)
    rad = math.fabs(alpha) * (math.pi / 180)

    h0, w0 = img.shape[:2]
    center = (w0 / 2, h0 / 2)

    h1 = h0 * math.cos(rad) + w0 * math.sin(rad)
    w1 = w0 * math.cos(rad) + h0 * math.sin(rad)

    ar = float(w0) / h0
    ar_rot = w1 / h1
    if ar < 1:
        t = float(w0) / ar_rot
    else:
        t = float(h0)

    Mr = cv2.getRotationMatrix2D(center, alpha, 1)
    shift_x = w1 / 2 - center[0]
    shift_y = h1 / 2 - center[1]
    Mr[0, 2] += shift_x
    Mr[1, 2] += shift_y

    img = cv2.warpAffine(img, Mr, (int(w1), int(h1)))

    h2 = t / (ar * math.sin(rad) + math.cos(rad))
    w2 = h2 * ar

    x, y = int(w1) / 2, int(h1) /2
    x -= int(w2 / 2)
    y -= int(h2 / 2)
    img = img[y:y + int(h2), x:x + int(w2)]
    return img


def rand_translate(img, low=-4, high=4):
    h, w = img.shape[:2]
    shift_x, shift_y = np.random.randint(low, high + 1, size=2)
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    img = cv2.warpAffine(img, M, (w, h))
    return img


def rand_scale(img, low=0.8, high=1.2):
    factor = np.random.uniform(low, high)
    method = cv2.INTER_LINEAR
    if factor < 1:
        method = cv2.INTER_AREA
    img = cv2.resize(img, None, fx=factor, fy=factor, interpolation=method)
    return img


def augment_img(img):
    img = np.transpose(img, (1, 2, 0))

    img = rand_scale(img)
    img = rand_translate(img)
    img = rand_rotate(img)

    img = np.transpose(img, (2, 0, 1))
    return img


def resize_img(img):
    img = np.transpose(img, (1, 2, 0))
    img = cv2.resize(img, (256, 256))
    h, w, _ = img.shape
    img = img[h//2-112:h//2+112, w//2-112:w//2+112]
    img = np.transpose(img, (2, 0, 1))
    return img


def augment_batch(X, resize_only=False):
    p = Pool()
    if not resize_only:
        X_new = p.map(augment_img, list(X))
    else:
        X_new = p.map(resize_img, list(X))
    p.terminate()
    X = X_new
    return X
