from __future__ import print_function
import numpy as np
import os
import sys
from scipy.interpolate import interp1d
from astropy.io import fits, ascii as asc
 
__all__ = ['CCD250', 'CCD290', 'COSMOS']

class detector:
    """
    Should contain general routines for generating detector properties.
    Note that the intent is for this to be passed to a *specfic*
    detector class later on."""
    def __init__(self):
        pass
        
        self.qe = None

        #get path for bundled package files
        self.bfile_dir = os.path.join(os.path.dirname(sys.modules['mavisetc'].__file__), 'data')
    
    def _read_qe(self):
        data = np.array(asc.read(os.path.join(self.bfile_dir, self.qefile)))
        twave = np.array(data['col1'])/1e3
        tqe = np.array(data['col2'])/100.
        return interp1d(twave, tqe, fill_value='extrapolate')


class CCD250(detector):
    name = 'CCD250'
    rn = 3. #e-/pixel/dit
    dark = 5.0/3600. #e-/pixel/s
    npix_det = 4004
    pixsize = 10e-3 #mm

    def __init__(self):
        #initialize the model base
        detector.__init__(self)
        
        self.qefile = 'E2V_QE_250.csv'
        self.qe_interp = self._read_qe()


class CCD290(detector):
    name = 'CCD290'
    rn = 3. #e-/pixel/dit
    dark = 4.0/3600. #e-/pixel/s
    npix_det = 9216
    pixsize = 10e-3 #mm

    def __init__(self):
        #initialize the model base
        detector.__init__(self)
        
        self.qefile = 'E2V_QE_290.csv'
        self.qe_interp = self._read_qe()
        

class COSMOS(detector):
    name = 'COSMOS'
    rn = 0.7 #e-/pixel/dit
    dark = 0.05 #e-/pixel/s
    npix_det = 6800
    pixsize = 10e-3 #mm

    def __init__(self):
        #initialize the model base
        detector.__init__(self)

        self.qefile = 'COSMOS.csv' 
        self.qe_interp = self._read_qe()

