from __future__ import print_function
import numpy as np
import os
import sys
from scipy.interpolate import interp1d
from astropy.io import fits, ascii as asc
 
__all__ = ['VLT']

class telescope:
    """
    Should contain general routines for a given telescope.
    Note that the intent is for this to be passed to a *specfic*
    telescope class later on."""
    def __init__(self):
        pass
    
    def _psf(self):
        pass


class VLT(telescope):
    name = 'VLT'
    area = 48.32507025
    def __init__(self):
        #initialize the model base
        telescope.__init__(self)

        #include some mirror stuff here so area is actually calculated
        #include PSF calculation as well?  could do

        #get path for bundled package files and read in mirror reflectivities
        bfile_dir = os.path.join(os.path.dirname(sys.modules['mavisetc'].__file__), 'data')
       
        #M1
        data = np.array(asc.read(os.path.join(bfile_dir, 'UT4_M1_reflect.csv')))
        twave1 = np.array(data['col1'])/1e3
        tref1 = np.array(data['col2'])/100.
        
        #M2
        data = np.array(asc.read(os.path.join(bfile_dir, 'UT4_M2_reflect.csv')))
        twave2 = np.array(data['col1'])/1e3
        tref2 = np.array(data['col2'])/100.
        
        #M3
        data = np.array(asc.read(os.path.join(bfile_dir, 'UT4_M3_reflect.csv')))
        twave3 = np.array(data['col1'])/1e3
        tref3 = np.array(data['col2'])/100.

        #generate "master" wavelength array based on individual input arrays
        wl_low = min(twave1.min(),twave2.min(),twave3.min())
        wl_high = max(twave1.max(),twave2.max(),twave3.max())

        self.telescope_wave = np.linspace(wl_low, wl_high, 10000)
        m1_reflect = interp1d(twave1, tref1, fill_value='extrapolate')(self.telescope_wave)
        m2_reflect = interp1d(twave2, tref2, fill_value='extrapolate')(self.telescope_wave)
        m3_reflect = interp1d(twave3, tref3, fill_value='extrapolate')(self.telescope_wave)

        #store throughput
        self.telescope_eff = m1_reflect*m2_reflect*m3_reflect


       
