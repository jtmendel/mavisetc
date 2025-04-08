from __future__ import print_function
import numpy as np
import os
import sys

from scipy.stats import norm

from ..telescopes import VLT

__all__ = ["lgs_source"]

class lgs_source():
    """
    Class to handle generating an LGS background spectrum. Can be computed
    for different return flux and Na altitude. Can be used to compute low, 
    medium, and high backgrounds.
    
    Output lgs radiance is in Photons/s/m^2/um/arcsec^2.
    """

    def __init__(self, altitude=100e3, return_flux=0.5e7, level='low', telescope=None, resolution=20000):
        """
        altitude is in meters
        return_flux is in photon/s/m2
        """
        self.nad_wavelength = 0.589 #microns

        #initialize the telescope
        if telescope is None:
            self.telescope = VLT()
        else:
            self.telescope = telescope()
            
        #miscelaneous
        self.small_num = 1e-70
        self.clight  = 2.997924580e18 #A/s
        self.h = 6.626196e-27 #Plancks constant in erg s

        #for computation
        self.altitude = altitude
        self.return_flux = return_flux
        self.level = level
        
        #some resolution business
        self.res = resolution
        self.step = 1./self.res/2. #set step sized

        #store initial FLI, airmass, pwv, and update internal parameters
        self.set_params()

    def set_params(self, altitude=None, return_flux=None, level=None):
        if altitude is not None:
            self.altitude = altitude
        if return_flux is not None:
            self.return_flux = return_flux
        if level is not None:
            self.level = level
        
        # Compute projected are of primary on the sky
        # Compute projected area of doughnut on the sky
        rdisk_out = 3600.*np.degrees(np.arctan(self.telescope.primary_radius/self.altitude)) # Outer disk radius in arcsec
        rdisk_in  = 3600.*np.degrees(np.arctan(self.telescope.secondary_radius/self.altitude)) # Inner obscuration radius in arcsec
        self.projected_area = np.pi*(rdisk_out**2 - rdisk_in**2) # disk area in arcsec^2

        #get scaling for "level" of background
        #'low' = no background
        #'medium' = single donut
        #'high' = where two donuts overlap
        if self.level not in ['low', 'medium', 'high']:
            raise ValueError('Input LGS level must be one of "low", "medium", or "high"')
        else:
            if self.level == 'low':
                flux_scale = 0.
            elif self.level == 'medium':
                flux_scale = 1.
            elif self.level == 'high':
                flux_scale = 2.
        
        # assume return flux is uniformly distributed over the projection of the mirror
        self.lgs_surface_brightness = flux_scale*self.return_flux / self.projected_area #Photons/s/m2/arcsec2

    def _make_line(self, wavelength, resolution):
        dpix = np.diff(wavelength)[0]
        edges = np.r_[wavelength-dpix/2., wavelength[-1]+dpix/2.]

        #convert line width to pixels
        #line_res = self.nad_wavelength * self.width / 2.998e5 / dpix #pixels
        inst_res = np.interp(self.nad_wavelength, wavelength, resolution) #pixels
        # line_sig = np.sqrt(line_res**2 + inst_res**2) #pixels
        line_sig = np.copy(inst_res) #assumes the NaD line is unresolved

        #build emission line template array
        nad_template = self.lgs_surface_brightness*np.diff(norm.cdf(edges, loc=self.nad_wavelength, scale=line_sig*dpix)) / dpix
        return nad_template

    
    def __call__(self, wavelength=None, resolution=None):
        
        if wavelength is None:
            raise ValueError("Wavelength must be provided for LGS background model")
        if resolution is None:
            resolution = np.zeros_like(wavelength)

        self.red_step = np.diff(wavelength)[0]
        self.res_pix = np.copy(resolution)
        self.red_wavelength = np.copy(wavelength)

        #generate line
        sim_nad = self._make_line(wavelength, resolution) #photons/s/cm2/arcsec/um
        
        return wavelength, sim_nad

