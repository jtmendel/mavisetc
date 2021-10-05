from __future__ import print_function
import numpy as np

from scipy.stats import norm
from ..filters import get_filter
from astropy.cosmology import FlatLambdaCDM

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

__all__ = ['flat_source']


class flat_source():
    """
    Class for defining source objects.  This just generates sources
    that are flat (constant) in either f_lambda or f_nu.
    """
    
    def __init__(self):
        
        #some conversion parameters
        self.small_num = 1e-70
        self.lsun = 3.839e33 #erg/s
        self.pc2cm = 3.08568e18 #pc to cm
        self.clight  = 2.997924580e18 #A/s
        self.h = 6.626196e-27 #Plancks constant in erg s

        #for correction to absolute mags
        self.mag2cgs = np.log10(self.lsun/4.0/np.pi/(self.pc2cm*self.pc2cm)/100.)
       
        self.wavelength = np.arange(300,30000,1)/1e4 #um

    def _set_filter(self, filt):
        #fetch filter transmission curves from FSPS
        #normalize and interpolate onto template grid

        #pull information for this filter
        fobj = get_filter(filt)
        fwl, ftrans = fobj.transmission
        
        ftrans = np.maximum(ftrans, 0.)
        trans_interp = np.asarray(np.interp(self.wavelength, fwl/1e4, 
                                  ftrans, left=0., right=0.), dtype=np.float)

        #normalize transmission
        ttrans = np.trapz(np.copy(trans_interp)/self.wavelength, self.wavelength)
        if ttrans < self.small_num: ttrans = 1.
        ntrans = np.maximum(trans_interp / ttrans, 0.0)
        
        self.transmission = ntrans
        return 

    def set_params(self, obs_mag=20., obs_band='sdss_r',
                   norm='point', units='f_lambda'):
        # flux should be in erg/s/cm^2
        # width should be in km/s
        """
        set parameters for a single emission line source
        """
        self.obs_mag = obs_mag
        self.obs_band = obs_band

        #set filter transmission
        self._set_filter(obs_band)
        
        #set normalization type
        self.norm_sb = False
        if norm == 'extended':
            self.norm_sb =  True

        #set units
        self.units = units

        return


    def _get_mag(self):
        #compute observed frame magnitudes or flux
        num = np.trapz(self.wavelength*self.flux*self.transmission, self.wavelength)
        denom = np.trapz(self.wavelength*self.transmission, self.wavelength)

        return -2.5*np.log10(num/denom) - 48.6
    

    def __call__(self, wavelength=None, **kwargs):
                
        self.red_step = self.step = np.diff(self.wavelength)[0]

        self.res_pix = np.ones_like(self.wavelength)
        self.red_wavelength = np.copy(self.wavelength)

        #generate dummy flat spectrum
        self.flux = np.full_like(self.wavelength, 1e-20) #erg/s/cm^2/hz

        if self.units == 'f_lambda': 
            #assume that the spectrum above is flat in f_lambda, convert to f_nu
            self.flux *= (self.wavelength*1e4)**2 / self.clight #now erg/s/cm^2/hz

        #measure in-band magnitude on the current template
        mag_scale = self._get_mag()

        #flux scaling to match requested magnitude
        self.flux_factor = 10**(-0.4*self.obs_mag)/10**(-0.4*mag_scale)

        #convert the spectrum to more useful units
        spec_scaled = np.copy(self.flux)*self.flux_factor #in erg/s/cm^2/hz

        photons = spec_scaled *100**2 / self.h / self.wavelength #photons/s/m^2/um.  If self.norm_sb then arcsec^-2
       
        return self.wavelength, photons

