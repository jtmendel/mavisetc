try:
    from ._version import __version__
except(ImportError):
    pass

from . import instruments
from . import sources
from . import sky
from . import utils
from . import telescopes
from . import detectors

__all__ = ['instruments', 'sources', 'sky', 'utils', 'telescopes', 'detectors']
