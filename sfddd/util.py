import logging
from time import time

import theano

logger = logging.getLogger(__name__)


def timed(f):
    def timer(*args, **kwargs):
        t0 = time()
        result = f(*args, **kwargs)
        tf = (time() - t0) / 60
        logger.info("--- %0.3f minutes ---" % tf)
        return result
    return timer


def gpu_free_mem(unit='gb'):
    """
    Returns free GPU memory

    unit : "gb" | "mb" | "kb" | "b"
    """
    info = theano.sandbox.cuda.basic_ops.cuda_ndarray.cuda_ndarray.mem_info()
    free_mem_b = float(info[0])
    valid_units = {'gb', 'mb', 'kb', 'b'}
    if unit == 'gb':
        ret = free_mem_b / 1024 / 1024 / 1024
    elif unit == 'mb':
        ret = free_mem_b / 1024 / 1024
    elif unit == 'kb':
        ret = free_mem_b / 1024
    elif unit == 'b':
        ret = free_mem_b
    else:
        raise ValueError("Parameter 'unit' not in %s" % valid_units)
    return ret
