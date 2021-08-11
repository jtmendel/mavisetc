from __future__ import print_function
import numpy as np
import fsps

from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

__all__ = ['fsps_source']


class fsps_source():
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
        
        #initialize SPS object
        self.sp = fsps.StellarPopulation(compute_vega_mags=False, zcontinuous=1,
                                         sfh=1, logzsol=0.0, dust_type=2, dust2=0.1, 
                                         imf_type=1, tau=1.0, add_neb_emission=1)
        
        #store master (rest-frame) wavelength array
        self.wavelength = self.sp.wavelengths / 1e4 #in microns
        self.step = np.r_[np.diff(self.wavelength),np.diff(self.wavelength)[-1]]
        self.red_wavelength = None
        self.red_step = None
        self.spectrum = None
        
        #resolution information
        self.res = 400 # just a guess(-ish) for BaSeL
        
        #hacky inclusion of MILES spectral resolution
        if len(self.wavelength) > 2000: #this catches when using MILES data
            self.res = np.ones(len(self.wavelength))*self.res
            miles_wave = (self.wavelength > 3525./1e4) & (self.wavelength < 7500./1e4)
            self.res[miles_wave] = (2.54/1e4)/self.wavelength[miles_wave]
                
        self.res_pix = self.wavelength / self.res / self.step / 2.355
        
        
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
       

    def set_params(self, age=3., tau=1., metallicity=0., 
                   redshift=1e-10, obs_mag=20., obs_band='sdss_r',
                   norm='point', sp_args={}, **kwargs):
        """
        valid args are those that match with parameters in 
        python-fsps
        """
        self.age = age
        self.redshift = redshift
        self.obs_mag = obs_mag
        self.obs_band = obs_band
        
        #set redshift dependent conversion factors
        self._set_redshift(redshift)
        
        #update FSPS parameters
        self.sp.params['tau'] = tau
        self.sp.params['logzsol'] = metallicity
           

        #pass any other dictionary values through to FSPS
        for key, value in sp_args.items():
            #print(key, value)
            #if key in self.sp.params:
            self.sp.params[key] = value

        #set normalization type
        self.norm_sb = False
        if norm == 'extended':
            self.norm_sb =  True

        return
                
    def __call__(self, **kwargs):
                
        #estimate in-band magnitude given the data provided
        mag_scale = self.sp.get_mags(tage=self.age, redshift=self.redshift, bands=[self.obs_band])
        self.flux_factor = 10**(-0.4*self.obs_mag)/10**(-0.4*mag_scale)

        #generate the initial spectrum given the provided parameters
        _, init_spec = self.sp.get_spectrum(tage=self.age)

        #convert the spectrum to more useful units
        spec_scaled = np.copy(init_spec)*self.flux_factor*self.fscale #in erg/s/cm^2/hz
        
        #photons = spec_scaled *100**2 / self.h / self.red_wavelength #photons/s/m^2/um.  If self.norm_sb then arcsec^-2
        photons = spec_scaled * 100**2 / self.h / self.red_wavelength #photons/s/m^2/um.  If self.norm_sb then arcsec^-2
        
        return self.red_wavelength, photons

