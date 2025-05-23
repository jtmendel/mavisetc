try:
    from ._version import __version__
except(ImportError):
    pass

from . import instruments
from . import sources
from . import background
from . import utils
from . import telescopes
from . import detectors
from . import filters

__all__ = ['instruments', 'sources', 'background', 'utils', 'telescopes', 'detectors', 'filters']
