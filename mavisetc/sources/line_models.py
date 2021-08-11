from __future__ import print_function
import numpy as np
from scipy.stats import norm

from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

__all__ = ['line_source']


class line_source():
    """
    Class for defining source objects.  For the moment this just 
    wraps generating an FSPS spectrum with a given magnitude, but 
    could reasonably be extended.
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
        
        
    def _set_redshift(self, redshift):
        #update internal working redshift
        self.redshift = redshift

        #update the wavelength array
        self.line_wavelength = (1.+redshift)*self.rest_wavelength

        #some derived parameters
        self.dm = cosmo.distmod(redshift).value - \
                2.5*np.log10(1+redshift) #redshift correction
        self.fscale = 10**(self.mag2cgs - 0.4*self.dm)
        return
       

    def set_params(self, flux=1e-16, wavelength=6564., 
                   redshift=1e-10, norm='point', width=30.,
                   **kwargs):
        # flux should be in erg/s/cm^2
        # width should be in km/s
        """
        set parameters for a single emission line source
        """
        self.flux = flux
        self.rest_wavelength = wavelength / 1e4
        self.width = width

        #set redshift dependent conversion factors
        self._set_redshift(redshift)
        
        #set normalization type
        self.norm_sb = False
        if norm == 'extended':
            self.norm_sb =  True

        return

    
    def _make_line(self, wavelength, resolution):
        dpix = np.diff(wavelength)[0]
        edges = np.r_[wavelength-dpix/2., wavelength[-1]+dpix/2.]
    
        #convert line width to pixels
        line_res = self.line_wavelength * self.width / 2.998e5 / dpix #pixels
        inst_res = np.interp(self.line_wavelength, wavelength, resolution) #pixels
        line_sig = np.sqrt(line_res**2 + inst_res**2) #pixels

        #build emission line template array
        emm_template = self.flux*np.diff(norm.cdf(edges, loc=self.line_wavelength, scale=line_sig*dpix)) / dpix
        return emm_template


    def __call__(self, wavelength=None, resolution=None):
                
        if wavelength is None:
            raise ValueError("Wavelength must be provided for line model")
        if resolution is None:
            resolution = np.zeros_like(wavelength)

        self.red_step = np.diff(wavelength)[0]
        self.res_pix = np.copy(resolution)
        self.red_wavelength = np.copy(wavelength)

        #generate line
        sim_spec = self._make_line(wavelength, resolution) #erg/s/cm^2/um

        #convert to useful units
        photons = sim_spec * 100**2 * wavelength / self.h / (self.clight/1e4) #photons/s/m^2/um


        #estimate in-band magnitude given the data provided
#        mag_scale = self.sp.get_mags(tage=self.age, redshift=self.redshift, bands=[self.obs_band])
#        flux_factor = 10**(-0.4*self.obs_mag)/10**(-0.4*mag_scale)
#
#        #generate the initial spectrum given the provided parameters
#        _, init_spec = self.sp.get_spectrum(tage=self.age)
#
#        #convert the spectrum to more useful units
#        spec_scaled = np.copy(init_spec)*flux_factor*self.fscale #in erg/s/cm^2/hz
        
#        photons = spec_scaled *100**2 / self.h / wavelength #photons/s/m^2/um.  If self.norm_sb then arcsec^-2
        
        return wavelength, photons

