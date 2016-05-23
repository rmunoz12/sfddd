import logging
from time import time

logger = logging.getLogger(__name__)


def timed(f):
    def timer(*args, **kwargs):
        t0 = time()
        result = f(*args, **kwargs)
        tf = (time() - t0) / 60
        logger.info("--- %0.3f minutes ---" % tf)
        return result
    return timer
