import os
import numpy as np
import glob
import sys

"""
Tools for working with the mavisetc filter set.

This is cribbed heavily from the FSPS filter implementation.
"""

__all__ = ["find_filter", "FILTERS", "get_filter", "list_filters"]


class Filter(object):

    def __init__(self, name, filename):
        self.name = name.lower()
        self.filename = filename
        self.trans_cache = None
        self.pivot = None

    def __str__(self):
        return "<Filter({0})>".format(self.name)

    def __repr__(self):
        return "<Filter({0})>".format(self.name)

    @property
    def lambda_eff(self):
        """Effective wavelength of Filter, in Angstroms."""
        #make sure the transmission informtion is loaded
        if self.trans_cache is None:
            self._load_transmission()

        if self.pivot is None:
            wl, tran = self.trans_cache
            self.pivot = np.sqrt(np.trapz(wl*tran, wl) / np.trapz(tran/wl, wl))

        return self.pivot

    @property
    def transmission(self):
        """Returns filter transmission: a tuple of wavelength (Angstroms) and
        an un-normalized transmission arrays.
        """
        if self.trans_cache is None:
            self._load_transmission()
        try:
            return self.trans_cache
        except KeyError as e:
            e.args += ("Could not find transmission data "
                       "for {0}".format(self.name))
            raise

    def _load_transmission(self):
        """Parse the allfilters.dat file into the TRANS_CACHE."""
        names = list_filters()
        lambdas, trans = [], []
        with open(self.filename) as f:
            for line in f:
                line.strip()
                if not line[0].startswith("#"):
                    try:
                        l, t = line.split()
                        lambdas.append(float(l))
                        trans.append(float(t))
                    except(ValueError):
                        pass
        self.trans_cache = (np.array(lambdas), 
                            np.array(trans))


def _load_filter_dict():
    """
    Load the filter list, creating a dictionary of :class:`Filter` instances.
    """
    
    #initialize filter directory
    filter_dir = os.path.join(
            os.path.dirname(sys.modules['mavisetc'].__file__), 'data/filters')

    filter_list = glob.glob('{0}/*.dat'.format(filter_dir))
    filters = {}
    for f in filter_list:
        _, fname = os.path.split(f)
        key, _ = os.path.splitext(fname)
        filters[key.lower()] = Filter(key, f)

    return filters


FILTERS = _load_filter_dict()


def find_filter(band):
    """
    Find the name for a filter.

    Usage:

    ::

        >>> import mavisetc
        >>> mavisetc.find_filter("u")
        ['johnson_u', 'stromgren_u', 'sdss_u']

    :param band:
        Something like the name of the band.

    """
    b = band.lower()
    possible = []
    for k in FILTERS.keys():
        if b in k:
            possible.append(k)
    return possible


def get_filter(name):
    """Returns the :class:`mavisetc.filters.Filter` instance associated with the
    filter name.

    :param name:
        Name of the filter, as found with :func:`find_filter`.
    """
    try:
        return FILTERS[name.lower()]
    except KeyError as e:
        e.args += ("Filter {0} does not exist. "
                   "Try using mavisetc.find_filter('{0}').".format(name),)
        raise


def list_filters():
    """Returns a list of all mavisetc filter names.
    """
    lst = [name for name in FILTERS.keys()]
    lst.sort(key=lambda x: x[0])
    return [l for l in lst]

#    return lst
