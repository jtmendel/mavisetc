from __future__ import print_function
import numpy as np
import os
import sys

import pandas as pd
import scipy as sp

from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
from ..filters import get_filter

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)


__all__ = ['etalon_source']


class etalon_source():
    """ FabryPerotEtalonSimulator provides a class to compute a Fabry-Perot interferometer cavity output. 
        The constructor has the following parameters:
        
            * lambda_vec is the array of lambda values in nm
            * mirror_separation is the separation between the mirrors in mm
            * mirror1_reflectivity is the reflectivity of the input mirror surface (could be an array over wavelength)
            * mirror2_reflectivity is the reflectivity of the output  mirror surface (could be an array over wavelength)
            * n_gap_material is the index of refraction of the material between the mirrors
            * angle_of_incidence is the angle in radians of the incident beam
            * clear_aperture is the aperture size of the 2-mirror system in mm
            * residual_power is the amount of maximum spherical curvature error between the two surfaces in nm
            * mirror_tilt_error is the error from parallel in radians between the two surfaces
            * rms_surface_irregularity is the amount of total peak to valley surface error between in nm
              the two mirror surfaces excluding power and tilt.
            * loss_coefficient is in parts per million
    """

    def __init__(self):

        #some relevant parameters
        self.f_num_factor = 0.0035
        self.int_sphere_factor = 4.7356e-04
        self.cal_unit_optics_transmission = 0.8
        pin_hole_size = 5.0 #5um pinhole in um
        sys_f_num = 15.0
        bessel_zero_pnt = 3.8317

        # physical constants
        self.clight = 299792458
        self.h = 6.62607004081e-34 #W s
        self.hes = 6.62607004081e-27 #erg ss
        self.clight_as = 2.99792458e18 #A/s

        #initialize directory for tabulated data
        self.bfile_dir = os.path.join(os.path.dirname(sys.modules['mavisetc'].__file__), 'data/etalon')


        eq_flux_csv_df = pd.read_csv(os.path.join(self.bfile_dir, "EQ99X_flux_100um_fibre.csv"))
        eq_flux_csv = eq_flux_csv_df.to_numpy()
        self.eq_flux_fcn = sp.interpolate.PchipInterpolator(eq_flux_csv[:,0],eq_flux_csv[:,1])
        #eq_flux_interpolated = eq_flux_fcn(lambda_vec)

        eq_reflect_csv_df = pd.read_csv(os.path.join(self.bfile_dir, "SLS_Reflectivity_Etatlon_80um.csv"))
        eq_reflect_csv = eq_reflect_csv_df.to_numpy()
        self.eq_reflect_fcn = sp.interpolate.PchipInterpolator(eq_reflect_csv[:,0],eq_reflect_csv[:,1])
        #eq_reflect_interpolated = eq_reflect_fcn(lambda_vec)

        #template parameters that will be populated later
        self.res = None
        self.res_pix = None
        self.wavelength = None
        self.red_wavelength = None
        self.red_step = None

        #filter parameters that will be populated later
        self.transmission = None




    def _airy_disc_fcn(I0,x):
        out_num = 2*sp.special.jv(1,x)
        out = np.divide(out_num,x)
        out = I0*np.multiply(out,out)
        return out


    def _airy_disc_integrated_power(P0,x):
        term1 = sp.special.jv(0,x)
        term2 = sp.special.jv(1,x)
        out = P0*(1-np.multiply(term1,term1)-np.multiply(term2,term2))
        return out


    def set_params(self, lambda_vec=np.linspace(370,940,100000), input_focal_ratio=10000.0, 
                 mirror_separation=0.08, mirror1_reflectivity=0.9, mirror2_reflectivity=0.9, 
                 n_of_gap_material=1.0, angle_of_incidence=0.0, clear_aperture=5.0, 
                 residual_power = 3.0, mirror_tilt_error=2.5e-7, rms_surface_irregularity = 1.5,
                 fibre_focal_ratio = 2.2, fibre_diameter=100, collimating_lens_focal_len=20.,
                 loss_coefficient = 50.0, norm='point', use_integrated_angle=True, num_radius_steps=100,
                 **kwargs):
        """ The set_etalon_parameters function sets or resets the etalon parameters.
            The parameters are the same as the constructor parameters."""
        
        self.lambda_vec = lambda_vec #nm
        self.input_focal_ratio = input_focal_ratio
        self.mirror_separation = mirror_separation
        self.mirror1_reflectivity = mirror1_reflectivity
        self.mirror2_reflectivity = mirror2_reflectivity
        self.n_of_gap_material = n_of_gap_material
        self.angle_of_incidence = angle_of_incidence
        self.clear_aperture = clear_aperture
        self.residual_power = residual_power
        self.mirror_tilt_error = mirror_tilt_error
        self.rms_surface_irregularity = rms_surface_irregularity
        self.loss_coefficient = loss_coefficient
        self.fibre_focal_ratio = fibre_focal_ratio
        self.fibre_diameter = fibre_diameter
        self.collimating_lens_focal_len = collimating_lens_focal_len
        self.num_radius_steps = num_radius_steps
        self.use_integrated_angle = use_integrated_angle
        self.fibre_radius = self.fibre_diameter*1e-3/2.
        self.max_angle_col = np.arctan(self.fibre_radius/self.collimating_lens_focal_len)
        self.eff_fl = 1.0/np.tan(self.max_angle_col)

        self.R = np.sqrt(self.mirror1_reflectivity*self.mirror2_reflectivity)
        self.FR = 4.0*self.R/((1-self.R)**2)

        #run calculation of etalon parameters
        self.compute_etalon_parameters()

        #set some template bits
        self.res = 100000

        self.wavelength = self.lambda_vec/1e3 #microns
        self.step = np.diff(self.wavelength)[0] #microns
        self.res_pix = self.wavelength / self.res / self.step / 2.355


        #set stuff for redshifting of the template
        self.red_step = self.step
        
        #set normalization type
        self.norm_sb = False
        if norm == 'extended':
            self.norm_sb =  True

        return


    def compute_etalon_parameters(self):
        """This is an internal function."""
        # compute the mirror finesse, Nr = 1/Nr_term        
        Nr_term = (1-self.R)/(np.pi*np.sqrt(self.R))       #1/finesse 

        # compute the plate defect finesse Nds
        Nds = np.divide(self.lambda_vec,2*self.residual_power)

        Nds_term_sq = np.reciprocal(np.multiply(Nds,Nds))

        # compute the second order surface defect finesse Ndg
        Ndg = np.divide(self.lambda_vec,4.7*self.rms_surface_irregularity)

        Ndg_term_sq = np.reciprocal(np.multiply(Ndg,Ndg))

        # compute the tilt defect finesse Ndp
            #change in separation of plates at edge of CA in nm
        delta_at_edge_of_aperture = (self.clear_aperture*1e6)*np.sin(self.mirror_tilt_error) 
        Ndp = np.divide(self.lambda_vec,np.sqrt(3)*delta_at_edge_of_aperture)

        Ndp_term_sq = np.reciprocal(np.multiply(Ndp,Ndp))

        Nd_term_sq = Nds_term_sq + Ndg_term_sq + Ndp_term_sq

        #Effects of non-collimated input
        self.theta_range = np.arctan(1./(2*self.input_focal_ratio))
        if not self.use_integrated_angle:
            sin_term = np.sin(self.theta_range/2.)
            Na_term = np.divide((self.mirror_separation*1e6)*4*sin_term*sin_term,self.lambda_vec)
        else:
            Na_term = 0.0
        Ne_term_sq = Nr_term*Nr_term+Nd_term_sq+Na_term*Na_term
        Ne_term = np.sqrt(Ne_term_sq)

        self.Ne = np.reciprocal(Ne_term)
        self.F = 4*np.multiply(self.Ne,self.Ne)/(np.pi*np.pi)

        return

    def _compute_etalon_curve_at_angle(self, angle):
        """Internal function"""
        trans_out_num = np.divide(self.F,self.FR)
        trans_out_den = 1.0 + self.F*(np.sin(np.divide(2*np.pi*self.n_of_gap_material*(self.mirror_separation*1.0e-3)*np.cos(angle),(self.lambda_vec*1e-9))))**2
        trans_out = np.divide(trans_out_num,trans_out_den)
        return trans_out

    def compute_etalon_curve(self):
        """ compute_etalon_curve computes and returns the etalon transmission curve.
            If a different set of lambda (wavelength) values, lambda_vec, is provided then the curve
            is updated using the new lambda (wavelength) values."""
            
      
        if not self.use_integrated_angle:
            trans_out = self._compute_etalon_curve_at_angle(self.angle_of_incidence)
        else:

            radius_vec = np.linspace(0,self.fibre_radius, self.num_radius_steps)
            dr = np.diff(radius_vec)[0]
            trans_out = np.zeros_like(self.lambda_vec)
            
            for radius in radius_vec:
                out_num = 4.0*np.pi*radius
                out_den = 2 + self.F - np.multiply(self.F,
                                                    np.cos(np.divide(4*np.pi*self.n_of_gap_material*(self.mirror_separation*1.0e-3), \
                                                    np.multiply(self.lambda_vec*1.0e-9,np.sqrt(1.0+radius*radius/(self.collimating_lens_focal_len))))))


                out = np.divide(out_num,out_den)                             
                trans_out = trans_out + out
            area = np.pi*self.fibre_radius*self.fibre_radius
            trans_out = np.divide(self.F,self.FR)*dr*np.divide(trans_out,area)

        return trans_out


#    def _set_template(self, norm='point'):
#        self.res = 100000
#
#        #ensure sampling onto a regular grid
#        wave_low, wave_high = temp_wave.min(), temp_wave.max()
#        npix = len(temp_wave)
#
#        self.wavelength = self.lambda_vec
#        self.step = np.diff(self.wavelength)[0]
#        self.res_pix = self.wavelength / self.res / self.step / 2.355
#
#        return
    

    def __call__(self, pinhole=False, **kwargs):
        #self.set_params(**kwargs)

        #compute the etalon parameters and curve
        self.etalon_curve = self.compute_etalon_curve() 

        #compute photon energy
        self.photon_energy_at_lambda = np.divide(self.h*self.clight,(self.lambda_vec*1e-9))
        self.eq_flux_interpolated = self.eq_flux_fcn(self.lambda_vec)
        self.eq_reflect_interpolated = self.eq_reflect_fcn(self.lambda_vec)

        self.etalon_at_mavis_focal_plane_mw = self.cal_unit_optics_transmission*self.int_sphere_factor\
                                            *self.f_num_factor*np.multiply(self.eq_flux_interpolated,
                                                                           self.etalon_curve)

        if pinhole: #compute the pinhole throughput
            self.diff_scaling_by_wavelength = np.divide(1.0,1.22*self.lambda_vec*self.sys_f_num*1e-3)
            self.airy_out = self._airy_disc_integrated_power(1.0,self.bessel_zero_pnt
                                                                 *self.diff_scaling_by_wavelength
                                                                 *self.pin_hole_size)
            self.etalon_at_mavis_focal_plane_mw = np.multiply(self.airy_out,
                                                                      self.etalon_at_mavis_focal_plane_mw)
            
        # interp_flux = np.interp(self.wavelength, temp_wave, temp_flux)/self.step #erg/s/cm^2/micron
        # self.template_flux = interp_flux * temp_wave**2 *1e4 / self.clight #in erg/s/cm^2/Hz

        etalon_in_erg_s_cm2_micron = (self.etalon_at_mavis_focal_plane_mw *1e9 / 1e3) / self.step #erg/s/cm^2/um
        template_flux = etalon_in_erg_s_cm2_micron * self.wavelength**2 * 1e4 / self.clight_as #erg/s/cm^2/Hz

        photons = template_flux *100**2 / self.hes / self.wavelength #photons/s/m^2/um.
        
        return self.wavelength, photons #photons/s/m^2/um.  If self.norm_sb then arcsec^-2

