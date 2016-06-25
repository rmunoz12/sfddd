"""
Builds `config` to be imported by training & prediction modules.

Modify default elements in this section as needed. Each sub-section header
identifies the attribute of config where the variables will be stored. Or,
alternatively, import classes at the end to roll your own.
"""

from collections import namedtuple
import logging

logger = logging.getLogger(__name__)


# config.*
model = 'inc'  # 'vgg' or 'inc'


# config.paths.*
data_folder = 'data/'
out_folder = 'out/'
cache_folder = 'cache/'

# ----------------------------------------------------------------------------

_PATHS = {'data_folder': data_folder,
          'out_folder': out_folder, 'cache_folder': cache_folder}

Paths = namedtuple('Paths', sorted(_PATHS))
_p = Paths(**_PATHS)

if model not in {'vgg', 'inc'}:
    logger.error('Unknown model: %s' % model)
    raise ValueError(model)

Config = namedtuple('Config', ['paths', 'model'])
config = Config(_p, model)
