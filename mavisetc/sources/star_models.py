from __future__ import print_function
import numpy as np
import fsps

from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

__all__ = ['stellar_source']


class stellar_source():
    """
    Class for defining source objects.  For the moment this just 
    wraps generating an FSPS spectrum with a given magnitude, but 
    could reasonably be extended.
    """
    
    def __init__(self, met=4):
        
        #some conversion parameters
        self.small_num = 1e-70
        self.lsun = 3.839e33 #erg/s
        self.pc2cm = 3.08568e18 #pc to cm
        self.clight  = 2.997924580e18 #A/s
        self.h = 6.626196e-27 #Plancks constant in erg s

        #for correction to absolute mags
        self.mag2cgs = np.log10(self.lsun/4.0/np.pi/(self.pc2cm*self.pc2cm)/100.)
        
        #initialize SPS object
        self.sp = fsps.StellarPopulation(zmet=met)
        
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
      

    def _set_redshift(self, redshift):
        #update internal working redshift
        self.redshift = redshift

        #update the wavelength array
        self.red_wavelength = self.wavelength
        self.red_step = self.step

        #some derived parameters
        self.dm = 0.
        self.fscale = 10**(self.mag2cgs - 0.4*self.dm)
        return
       

    def set_params(self, mact, logt, lbol, logg, phase, peraa=False,
                   comp=0, obs_mag=20., obs_band='sdss_r',
                   norm='point', **kwargs):
        """
        valid args are those that match with parameters in 
        python-fsps
        """

        self.mact = mact
        self.logt = logt
        self.lbol = lbol
        self.logg = logg
        self.phase = phase
        self.peraa = peraa
        self.comp = comp
        
        self.obs_mag = obs_mag
        self.obs_band = obs_band

        #set redshift dependent conversion factors
        self._set_redshift(0.)
        
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

        #pull stellar template
        self.template_spec = self.sp._get_stellar_spectrum(self.mact, self.logt, self.lbol, self.logg, 
                                                           self.phase, self.comp, peraa=False)
        self.template_flux = self.template_spec * self.fscale

        #get current magnitude
        mag_scale = self._get_mag()
        self.flux_factor = 10**(-0.4*self.obs_mag)/10**(-0.4*mag_scale)

        #convert the spectrum to more useful units
        spec_scaled = np.copy(self.template_spec)*self.flux_factor*self.fscale #in erg/s/cm^2/hz
        
        photons = spec_scaled * 100**2 / self.h / self.red_wavelength #photons/s/m^2/um.  If self.norm_sb then arcsec^-2
        
        return self.red_wavelength, photons

