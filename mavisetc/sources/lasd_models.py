from __future__ import print_function
import numpy as np
import os
import sys

from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
from ..filters import get_filter

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

__all__ = ['lasd_source']

class lasd_source():
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
        
        #initialize the LASD template dictionary
        self.bfile_dir = os.path.join(os.path.dirname(sys.modules['mavisetc'].__file__), 'data/lasd_templates')
      
        #open the master file and parse the names, flux, redshift
        self.template_dict = {}
        with open(os.path.join(self.bfile_dir, 'lasd_measurements.cat'), 'r') as sfile:
            for line in sfile:
                if line[0] != '#':
                    temp = line.strip().split(None)
                    #stores full spectral path, log(luminosity), redshift
                    self.template_dict[temp[0]] = (os.path.join(self.bfile_dir,'{0}.ascii'.format(temp[0])), 
                                                   float(temp[1]), float(temp[2]))

        #template parameters that will be populated later
        self.res = 17000#500
        self.res_pix = None
        self.wavelength = None
        self.red_wavelength = None
        self.red_step = None

        #filter parameters that will be populated later
        self.transmission = None
        self.type = None

    def templates(self):
        return self.template_dict.keys()
        
    def _set_redshift(self, redshift):
        #update internal working redshift
        self.redshift = redshift

        #update the wavelength array
        self.red_wavelength = self.wavelength * (1.+redshift)/(1+self.template_redshift)
        self.red_step = self.step * (1.+redshift)/(1+self.template_redshift) #pixel size in um

        #some derived parameters
        self.dm = cosmo.distmod(redshift).value - \
                2.5*np.log10(1+redshift) #redshift correction
        self.fscale = 10**(self.mag2cgs - 0.4*self.dm)
        return
      
#    def _set_filter(self, filt):
#        #fetch filter transmission curves
#        #normalize and interpolate onto template grid
#
#        #pull information for this filter
#        fobj = get_filter(filt)
#        fwl, ftrans = fobj.transmission
#        
#        ftrans = np.maximum(ftrans, 0.)
#        trans_interp = np.asarray(interp(self.red_wavelength, fwl/1e4, 
#                                  ftrans, left=0., right=0.), dtype=float)
#
#        #normalize transmission
#        ttrans = np.trapz(np.copy(trans_interp)/self.red_wavelength, self.red_wavelength)
#        if ttrans < self.small_num: ttrans = 1.
#        ntrans = np.maximum(trans_interp / ttrans, 0.0)
#        
#        self.transmission = ntrans
#        return 

    def _set_template(self, template):
        if template not in self.template_dict:
            raise("Template must be one of {0}".format(','.join(list(self.templates()))))

        itemp_wave, itemp_flux = [], []
        with open(self.template_dict[template][0],'r') as tfile:
            for line in tfile:
                if line[0] != '#':
                    temp = line.strip().split(None) 
                    itemp_wave.append(float(temp[0])) #microns
                    itemp_flux.append(float(temp[1]))
        temp_flux = np.asarray(itemp_flux, dtype=float)
        temp_wave = np.asarray(itemp_wave, dtype=float)


        #for correction to absolute mags
        self.mag2cgs = np.log10(1/4.0/np.pi/(self.pc2cm*self.pc2cm)/100.)

        self.wavelength = temp_wave/1e4 #in um
        self.step = np.diff(self.wavelength)[0]
        self.res_pix = self.wavelength / self.res / self.step / 2.355
        self.template_flux = temp_flux * temp_wave**2 / self.clight #in erg/s/cm^2/Hz
        self.template_lum = self.template_dict[template][1]
        self.template_redshift = self.template_dict[template][2]

        self.template_dm = cosmo.distmod(self.template_redshift).value - \
                2.5*np.log10(1+self.template_redshift) #redshift correction
        #self.template_fscale = 10**(self.mag2cgs - 0.4*self.template_dm)
        self.template_fscale = 10**(self.mag2cgs - 0.4*self.template_dm)

        return 
        

    def set_params(self, template=None, lya_lum=None, redshift=1e-10,
                   norm='point', **kwargs):
        """
        Needs information
        """
        self.redshift = redshift
        self.lya_lum = lya_lum
        
        #load template here
        self._set_template(template)

        #set redshift dependent conversion factors
        self._set_redshift(redshift)
       
#        #set filter transmission
#        self._set_filter(obs_band)
        self.flux_scale = 10**lya_lum / 10**self.template_lum

        #set normalization type
        self.norm_sb = False
        if norm == 'extended':
            self.norm_sb =  True

        return
 
#    def _get_mag(self):
#        #compute observed frame magnitudes or flux
#        num = np.trapz(self.red_wavelength*self.template_flux*self.transmission, self.red_wavelength)
#        denom = np.trapz(self.red_wavelength*self.transmission, self.red_wavelength)
#
#        return -2.5*np.log10(num/denom) - 48.6
               
    def __call__(self, **kwargs):
               
        #flux scaling to match requested magnitude
        self.flux_factor = self.fscale / self.template_fscale

        #convert the spectrum to more useful units
        spec_scaled = np.copy(self.template_flux)*self.flux_factor * self.flux_scale #in erg/s/cm^2/A
       
        photons = spec_scaled  *100**2 / self.h / self.red_wavelength#* 100**2 / self.h / self.red_wavelength #photons/s/m^2/um.  If self.norm_sb then arcsec^-2
        
        return self.red_wavelength, photons

