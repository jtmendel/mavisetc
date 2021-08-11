from __future__ import print_function
import numpy as np
import os
import sys
import fsps
from astropy.io import fits

from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

__all__ = ['template_source', 'stellar_source']

class template_source():
    """
    Class for defining source objects.  This just pulls in fits files
    from the Kinney and Calzetti template library.
    """
    
    def __init__(self):
        
        #some conversion parameters
        self.small_num = 1e-70
        self.lsun = 3.839e33 #erg/s
        self.pc2cm = 3.08568e18 #pc to cm
        self.clight  = 2.997924580e18 #A/s
        self.h = 6.626196e-27 #Plancks constant in erg s

        #for correction to absolute mags
        self.mag2cgs = np.log10(1/4.0/np.pi/(self.pc2cm*self.pc2cm)/100.)
        
        #initialize template dictionary
        self.bfile_dir = os.path.join(os.path.dirname(sys.modules['mavisetc'].__file__), 'data/kc_templates')
       
        self.template_dict = {'E':'elliptical_template.fits',
                             'S0': 's0_template.fits',
                             'Sa': 'sa_template.fits',
                             'Sb': 'sb_template.fits',
                             'Sc': 'sc_template.fits',
                             'SB1': 'starb1_template.fits',
                             'SB2': 'starb2_template.fits',
                             'SB3': 'starb3_template.fits',
                             'SB4': 'starb4_template.fits',
                             'SB5': 'starb5_template.fits',
                             'SB6': 'starb6_template.fits'
                             }

        #template parameters that will be populated later
        self.res = 400
        self.res_pix = None
        self.wavelength = None
        self.red_wavelength = None
        self.red_step = None

        #filter parameters that will be populated later
        self.transmission = None


    def templates(self):
        return self.template_dict.keys()
        
    def _set_redshift(self, redshift):
        #update internal working redshift
        self.redshift = redshift

        #update the wavelength array
        self.red_wavelength = self.wavelength * (1.+redshift)
        self.red_step = self.step * (1.+redshift) #pixel size in um

        #some derived parameters
        self.dm = cosmo.distmod(redshift).value - \
                2.5*np.log10(1+redshift) #redshift correction
        self.fscale = 10**(self.mag2cgs - 0.4*self.dm)
        return
      
    def _set_filter(self, filt):
        #fetch filter transmission curves from FSPS
        #normalize and interpolate onto template grid

        #lookup for filter number given name
        fsps_filts = fsps.list_filters()
        filt_lookup = dict(zip(fsps_filts, range(1,len(fsps_filts)+1)))

        #reference in case given a spitzer or mips filter...probably not an issue right now.
        mips_dict = {90:23.68*1e4, 91:71.42*1e4, 92:155.9*1e4}
        spitzer_dict = {53:3.550*1e4, 54:4.493*1e4, 55:5.731*1e4, 56:7.872*1e4}
        
        #pull information for this filter
        fobj = fsps.get_filter(filt)
        filter_num = filt_lookup[filt]
        
        fwl, ftrans = fobj.transmission
        
        ftrans = np.maximum(ftrans, 0.)
        trans_interp = np.asarray(np.interp(self.red_wavelength, fwl/1e4, 
                                  ftrans, left=0., right=0.), dtype=np.float)

        #normalize transmission
        ttrans = np.trapz(np.copy(trans_interp)/self.red_wavelength, self.red_wavelength)
        if ttrans < self.small_num: ttrans = 1.
        ntrans = np.maximum(trans_interp / ttrans, 0.0)
        
        if filter_num in mips_dict:
            td = np.trapz(((self.red_wavelength/mips_dict[filter_num])**(-2.))*ntrans/self.red_wavelength, self.red_wavelength)
            ntrans = ntrans/max(1e-70,td)

        if filter_num in spitzer_dict:
            td = np.trapz(((self.red_wavelength/spitzer_dict[filter_num])**(-1.0))*ntrans/self.red_wavelength, self.red_wavelength)
            ntrans = ntrans/max(1e-70,td)

        self.transmission = ntrans
        return 


    def _set_template(self, template):
        if template not in self.template_dict:
            raise("Template must be one of {0}".format(','.join(list(self.templates()))))

        with fits.open(os.path.join(self.bfile_dir, self.template_dict[template])) as ffile:
            tab = ffile[1].data
            temp_flux = np.array(tab.FLUX, dtype=np.float)
            temp_wave = np.array(tab.WAVELENGTH, dtype=np.float)

        self.wavelength = temp_wave/1e4 #in um
        self.step = np.diff(self.wavelength)[0]
        self.res_pix = self.wavelength / self.res / self.step / 2.355
        self.template_flux = temp_flux * temp_wave**2 / self.clight #in erg/s/cm^2/Hz
        return 
        

    def set_params(self, template=None, 
                   redshift=1e-10, obs_mag=20., obs_band='sdss_r',
                   norm='point', **kwargs):
        """
        valid args are those that match with parameters in 
        python-fsps
        """
        self.redshift = redshift
        self.obs_mag = obs_mag
        self.obs_band = obs_band
        
        #load template here
        self._set_template(template)

        #set redshift dependent conversion factors
        self._set_redshift(redshift)
       
        #set filter transmission
        self._set_filter(obs_band)

        #set normalization type
        self.norm_sb = False
        if norm == 'extended':
            self.norm_sb =  True

        return
 
    def _get_mag(self):
        #compute observed frame magnitudes or flux
        num = np.trapz(self.red_wavelength*self.template_flux*self.transmission, self.red_wavelength)
        denom = np.trapz(self.red_wavelength*self.transmission, self.red_wavelength)

        return -2.5*np.log10(num/denom) - 48.6
               
    def __call__(self, **kwargs):
               
        #measure in-band magnitude on the current template
        mag_scale = self._get_mag()

        #flux scaling to match requested magnitude
        self.flux_factor = 10**(-0.4*self.obs_mag)/10**(-0.4*mag_scale)

        #convert the spectrum to more useful units
        spec_scaled = np.copy(self.template_flux)*self.flux_factor #in erg/s/cm^2/hz
        
        photons = spec_scaled *100**2 / self.h / self.red_wavelength #photons/s/m^2/um.  If self.norm_sb then arcsec^-2
        
        return self.red_wavelength, photons


class stellar_source():
    """ A class to handle stellar template source
    """
    def __init__(self):
        pass

