from __future__ import print_function
import numpy as np
from scipy.stats import norm
import os
import sys

__all__ = ['lamp_source']


class lamp_source():
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
        self.mag2cgs = np.log10(1/4.0/np.pi/(self.pc2cm*self.pc2cm)/100.)

        #initialize template dictionary
        self.bfile_dir = os.path.join(os.path.dirname(sys.modules['mavisetc'].__file__), 'data/lamp_templates')

        self.template_dict = {'Ne': 'Hector_Ne_to_Focus.csv',
                              'Xe': 'Hector_Xe_to_Focus.csv',
                              'Cd': 'Cd_to_Focus.csv',
                              'Zn': 'Zn_to_Focus.csv',
                              'Etalon': 'LDLS_100um_Core_to_Focus_Etalon.csv',
                              'Flat_QTH': 'Thorlabs_SLS201L_QTH_to_Focus.csv',
                              'Flat_QTH_alt': 'Thorlabs_OSL2IR_QTH_to_Focus.csv',
                              'Flat_LDLS': 'LDLS_100um_Core_to_Focus.csv',
                              'Pinhole_QTH': 'Thorlabs_SLS201L_QTH_Fibre_to_Focus_With_Spectrograph_Grid_Pinholes.csv',
                              'Pinhole_QTH_alt': 'Thorlabs_OSL2IR_QTH_Fibre_to_Focus_With_Spectrograph_Grid_Pinholes.csv',
                              'Pinhole_LDLS': 'LDLS_100um_Core_Fibre_to_Focus_With_Spectrograph_Grid_Pinholes.csv',
                              'ACM': 'ACM_Pinhole_MGG_Lamps.csv'
                             }

        self.lamp_scale = {'Ne': 2,
                           'Xe': 2,
                           'Cd': 2,
                           'Zn': 2,
                           'Etalon': 1,
                           'Flat_QTH': 1,
                           'Flat_QTH_alt': 1,
                           'Flat_LDLS': 1,
                           'Pinhole_QTH': 1,
                           'Pinhole_QTH_alt': 1,
                           'Pinhole_LDLS': 1,
                           'ACM': 1,
                           }

        self.template_dist = {'Ne': 'extended',
                              'Xe': 'extended',
                              'Cd': 'extended',
                              'Zn': 'extended',
                              'Etalon': 'extended',
                              'Flat_QTH': 'extended',
                              'Flat_QTH_alt': 'extended',
                              'Flat_LDLS': 'extended',
                              'Pinhole_QTH': 'pinhole',
                              'Pinhole_QTH_alt': 'pinhole',
                              'Pinhole_LDLS': 'pinhole',
                              'ACM': 'pinhole'
                             }


        #template parameters that will be populated later
        self.res = None
        self.res_pix = None
        self.wavelength = None
        self.red_wavelength = None
        self.red_step = None

        #filter parameters that will be populated later
        self.transmission = None
        self.type = 'lamp'

    def templates(self):
        return self.template_dict.keys()

    def _set_template(self, template):
        if template not in self.template_dict:
            raise("Template must be one of {0}".format(','.join(list(self.templates()))))

        #store the current template internally
        self.template = template
        self.template_norm = self.template_dist[template] #treat pinholes differently

        #these are the line wavelengths and fluxes
        temp_flux = []
        temp_wave = []
        
        #for pinholes, read the integrated flux
        if self.template_norm == 'pinhole':
            read_col = 1
            flux_scale = 1e7
        else:
            read_col = 2
            flux_scale = 1e9
        with open(os.path.join(self.bfile_dir, self.template_dict[template]),'r') as ffile:
            for line in ffile:
                if line[0] != '#': #ignore comments
                    temp = line.strip().split(',')
                    temp_wave.append(float(temp[0])/1e3) #conversion from nm to microns
                    temp_flux.append(float(temp[read_col])*flux_scale) 
                    #includes conversion from W/mm^2/line to erg/s/cm^2/line (for arcs). 
                    #For the etalon/flats this is erg/s/cm^2/nm.
                    #For the pinholes this is erg/s/nm
                                                                
        self.line_flux = self.lamp_scale[template]*np.asarray(temp_flux)
        self.line_wave = np.asarray(temp_wave)

        return

    def set_params(self, template=None, **kwargs):
        """
        Needs information
        """

        #load template here
        self._set_template(template)

        #set normalization type
        self.norm_sb = False

        return


    def _make_line(self, line_wave, line_flux, wavelength, resolution):
        dpix = np.diff(wavelength)[0]
        edges = np.r_[wavelength-dpix/2., wavelength[-1]+dpix/2.]
    
        #convert line width to pixels
        inst_res = np.interp(line_wave, wavelength, resolution) #pixels

        #build emission line template array
        emm_template = line_flux*np.diff(norm.cdf(edges, loc=line_wave, scale=inst_res*dpix)) / dpix
        return emm_template
   

    def _make_spectrum(self, wavelength, resolution):
        #build the emission spectrum at instrument resolution
        spec_out = np.zeros_like(wavelength)
        for lwave, lflux in zip(self.line_wave, self.line_flux):
            spec_out += self._make_line(lwave, lflux, wavelength, resolution)    
        return spec_out


    def __call__(self, wavelength=None, resolution=None):
                
        if wavelength is None:
            raise ValueError("Wavelength must be provided for lamp model")
        if resolution is None:
            raise ValueError("Resolution must be provided for lamp model")

        self.red_step = np.diff(wavelength)[0]
        if self.template in ['Etalon']:
            self.res_pix = np.zeros_like(wavelength)
        else:
            self.res_pix = np.copy(resolution)
        self.red_wavelength = np.copy(wavelength)

        #generate line
        if self.template in ['Ne', 'Xe', 'Cd', 'Zn']: #these are just line lists, need to generate spectrum
            sim_spec = self._make_spectrum(wavelength, resolution) #erg/s/cm^2/um
        else: #for the other lamps (etalon, QTH, LDLS), treat them as spectra
            sim_spec = np.interp(wavelength, self.line_wave, self.line_flux) * 1e3 #erg/s/cm^2/um or erg/s/um

        #convert to useful units
        if self.template_norm == 'pinhole':
            photons = sim_spec * wavelength / self.h / (self.clight/1e4) #photons/s/um
        else: 
            photons = sim_spec * 100**2 * wavelength / self.h / (self.clight/1e4) #photons/s/m^2/um

        return wavelength, photons





