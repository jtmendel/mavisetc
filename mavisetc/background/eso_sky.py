from __future__ import print_function
import numpy as np
import skycalc_cli
import json
import os
import sys

from astropy.io import fits
from ..filters import get_filter

__all__ = ["sky_source"]

class sky_source():
    """
    Class to handle generating a sky spectrum using the ESO advanced 
    sky model.  Sky can be computed for different FLI, airmass, and PWV.
    
    Output sky radiance is in Photons/s/m^2/um/arcsec^2.
    """
    def __init__(self, fli=0.5, airmass=1.2, pwv=5, 
                resolution=20000, offline=False, wavelength='ir'):
        
        #for computing magnitudes
        self.small_num = 1e-70
        self.clight  = 2.997924580e18 #A/s
        self.h = 6.626196e-27 #Plancks constant in erg s
        
        self.thermal_dict = {'opt':{'therm_t1':285., 'therm_e1':0.20,
                                   'therm_t2':0.0, 'therm_e2':0.0,
                                   'therm_t3':33., 'therm_e3':0.01},
                        'ir':{'therm_t1':283., 'therm_e1':0.20,
                                    'therm_t2':125., 'therm_e2':0.1,
                                    'therm_t3':40., 'therm_e3':0.01}
                        }


        self.res = resolution
        self.step = 1./self.res/2. #set step sized based on resolution at 1um
        
        self.paramDict = {'airmass': 1.0,
                          'pwv_mode':  'pwv',
                          'season': 0,
                          'time': 0,
                          'pwv': 3.5,
                          'msolflux': 130.0,
                          'incl_moon':'Y',
                          'moon_sun_sep': 90.0,
                          'moon_target_sep': 45.0,
                          'moon_alt': 45.0,
                          'moon_earth_dist': 1.0,
                          'incl_starlight': 'Y',
                          'incl_zodiacal': 'Y',
                          'ecl_lon': 135.0,
                          'ecl_lat': 90.0,
                          'incl_loweratm': 'Y',
                          'incl_upperatm': 'Y',
                          'incl_airglow': 'Y',
                          'incl_therm': 'Y',
                          'therm_t1': 287.0,
                          'therm_e1': 0.2,
                          'therm_t2': 0.0,
                          'therm_e2': 0.0,
                          'therm_t3': 33.0,
                          'therm_e3': 0.01,
                          'vacair': 'vac',
                          'wmin': 300.0,
                          'wmax': 3500.0,
                          'wgrid_mode': 'fixed_wavelength_step',
                          'wdelta': self.step*1e3,
                          'wres': self.res,
                          'lsf_type': 'none',
                          'lsf_gauss_fwhm': 1.0,
                          'lsf_boxcar_fwhm': 0.,
                          'observatory': 'paranal'}
       
        for key, val in self.thermal_dict[wavelength].items():
            self.paramDict[key] = val 

        #storage params
        self.update_sky = False
        self.emm = None
        self.trans = None
        self.wavelength = None
        self.fli = None
        self.airmass = None
        self.pwv = None

        #store initial FLI, airmass, pwv, and update internal parameters
        self.set_params(fli=fli, airmass=airmass, pwv=pwv)

        #reference sky information
        self.ref_dir = os.path.join(os.path.dirname(sys.modules['mavisetc'].__file__), 'data')
        self.ref_skies = ['ref_sky_dark.fits','ref_sky_grey.fits','ref_sky_bright.fits']
   
        #check if running in offline mode
        self.offline = offline


    def _set_filter(self, filt, wavelength):
        #fetch filter transmission curves from FSPS
        #normalize and interpolate onto template grid

        #pull information for this filter
        fobj = get_filter(filt)
        fwl, ftrans = fobj.transmission
        
        ftrans = np.maximum(ftrans, 0.)
        trans_interp = np.asarray(np.interp(wavelength, fwl/1e4, 
                                  ftrans, left=0., right=0.), dtype=np.float)

        #normalize transmission
        ttrans = np.trapz(np.copy(trans_interp)/wavelength, wavelength)
        if ttrans < self.small_num: ttrans = 1.
        ntrans = np.maximum(trans_interp / ttrans, 0.0)
        
        self.transmission = ntrans
        return 


    def _get_moon_sun_sep(self, fli=None):
        #convert FLI to moon sun separation
        return 180.-np.degrees(np.arccos(2*np.clip(fli,0.01,0.99)-1))
        
    def set_params(self, fli=None, airmass=None, pwv=None, obs_mag=None, obs_band='johnson_v'):
        #store values and update parameter dictionary
        if fli is not None:
            if fli != self.fli: #is this different than the current value:?
                sep = self._get_moon_sun_sep(fli)
                self.paramDict['moon_sun_sep'] = sep
                self.fli = fli
                self.update_sky = True
        if airmass is not None:
            if airmass != self.airmass: #is this different than the current value:?
                self.paramDict['airmass'] = airmass
                self.airmass = airmass
                self.update_sky = True
        if pwv is not None:
            if pwv != self.pwv: #is this different than the current value:?
                self.paramDict['pwv'] = pwv
                self.pwv = pwv
                self.update_sky = True
        
        self.obs_mag = obs_mag
        self.obs_band = obs_band
               
    def _read_sky(self, skyfile):
        with fits.open(skyfile) as sfile:
            tab = sfile[1].data
            self.wavelength = np.asarray(tab['LAM'], dtype=np.float) / 1e3
            self.trans = np.asarray(tab['TRANS'], dtype=np.float)
            self.emm = np.asarray(tab['FLUX'], dtype=np.float)
      
        #update resolution in pixels
        self.res_pix = self.wavelength / self.res / self.step / 2.355


    def _scale_sky(self):
        if self.obs_band is not None:
            self._set_filter(self.obs_band, self.wavelength)

        emm_cgs = np.copy(self.emm)*self.h*self.wavelength / 100**2 #erg/s/cm^2/Hz/arcsec^2
        num = np.trapz(self.wavelength*emm_cgs*self.transmission, self.wavelength)
        denom = np.trapz(self.wavelength*self.transmission, self.wavelength)
            
        self.default_sky_mag = -2.5*np.log10(num/denom) - 48.6

        if self.obs_mag is not None: #compute scale factor for the sky
            #emm_cgs = np.copy(self.emm)*self.h*self.wavelength / 100**2 #erg/s/cm^2/Hz/arcsec^2
            #num = np.trapz(self.wavelength*emm_cgs*self.transmission, self.wavelength)
            #denom = np.trapz(self.wavelength*self.transmission, self.wavelength)
            
            #mag = -2.5*np.log10(num/denom) - 48.6
            scale = 10**(-0.4*self.obs_mag)/10**(-0.4*self.default_sky_mag)
            print("sky scaling:", scale, mag, self.obs_mag)
        else:
            scale = 1.

        #adjust scaling
        return scale


    def _calc_sky(self):
        #actually run the sky model
        parfile = 'sky_pars.json'
        skyfile = 'sky_spec.fits'

        #dump default parameters
        with open(parfile, 'w') as json_file:
            json.dump(self.paramDict, json_file)
        
        #run the skycalc call
        os.system('skycalc_cli -i {0} -o {1}/{2}'.format(parfile, os.getcwd(), skyfile))
        
        #pull in the sky spectrum
        self._read_sky(skyfile)
       
        os.remove(parfile)
        os.remove(skyfile)
        self.update_sky = False
            
    def __call__(self, fli=None, airmass=None, pwv=None):
        self.set_params(fli=fli, airmass=airmass, pwv=pwv, obs_band=self.obs_band, obs_mag=self.obs_mag)
        
        if self.update_sky:
            if self.offline:
                mindex = np.argmin(np.fabs(self.fli - np.array([0.0,0.5,1.0])))
                sky_use = os.path.join(self.ref_dir, self.ref_skies[mindex])
                self._read_sky(sky_use)
            else:
                try: #try to execute the command line interface
                    self._calc_sky()
                except: #not connected to the internet? Find the closest FLI from the reference
                    mindex = np.argmin(np.fabs(self.fli - np.array([0.0,0.5,1.0])))
                    sky_use = os.path.join(self.ref_dir, self.ref_skies[mindex])
                    self._read_sky(sky_use)

        #get scaling (if any)
        scale = self._scale_sky()

        return self.wavelength, self.emm*scale, self.trans
