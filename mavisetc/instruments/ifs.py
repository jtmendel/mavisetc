from __future__ import print_function
from astropy.io import fits, ascii as asc
from astropy.convolution import (convolve_fft, Gaussian2DKernel)
from scipy.interpolate import interp1d
from scipy.stats import multivariate_normal as mvn

import numpy as np
import os
import glob
import sys
import warnings

from ..utils.smoothing import smooth

#import some bits for the instruments
from ..detectors import CCD290
from ..telescopes import VLT

__all__ = ["MAVIS_IFS"]


class IFSInstrument:
    """
    Generic instrument class with some generic instrument routines.  
    """
    def __init__(self):
        self.source_obs = None
        self.sky_obs = None
        self.detector = None
        self.telescope = None
        self.tpt_includes_telescope = None


    def _EE_gaussian(self, seeing, binning=1, **kwargs):
        #Assumes provided seeing is in the v-band
        sigma_pix = seeing / self.pix_scale / 2.355 

        wave_temp = np.linspace(self.inst_wavelength.min(), self.inst_wavelength.max(), 50)
        sig_temp = sigma_pix * (wave_temp/.500)**(-1./5.)
        ee_temp = np.zeros_like(wave_temp)
        for ii, tsig in enumerate(sig_temp):
            ee_temp[ii] = Gaussian2DKernel(tsig, x_size=binning, y_size=binning,
                                            mode='oversample', factor=100).array.sum()

        ee_array = interp1d(wave_temp, ee_temp, kind='quadratic')(self.inst_wavelength)

        return ee_array, binning**2    


    def make_sky_spectrum(self, sky):
        sky_wave, sky_emm, sky_trans = sky()
        
        match_res_sky = np.interp(sky_wave, self.inst_wavelength, self.res_pix,
                                 left=self.res_pix[0], right=self.res_pix[-1])
        offset_res_sky = np.sqrt(np.clip((match_res_sky*self.step/sky.step)**2 - 
                                         sky.res_pix**2, 1e-10, None))
        conv_emm = smooth(sky_emm, offset_res_sky)
        conv_trans = smooth(sky_trans, offset_res_sky)
                    
        #resample onto output grid
        sky_emm_resampled = np.interp(self.inst_wavelength, sky_wave, conv_emm)
        sky_trans_resampled = np.clip(np.interp(self.inst_wavelength, 
                                                sky_wave, conv_trans),0,1)

        return sky_emm_resampled, sky_trans_resampled


    def make_source_spectrum(self, source):
        #generate source spectrum
        source_wave, source_phot = source(wavelength=self.inst_wavelength, resolution=self.res_pix)
        
        #convolve source to instrument resolution (if higher) or else leave be
        match_res_source = np.interp(source_wave, self.inst_wavelength, self.res_pix,
                                     left=self.res_pix[0], right=self.res_pix[-1]) #instrument resolution on source wavelength grid
        offset_res_source = np.sqrt(np.clip((match_res_source*self.step/source.red_step)**2 - 
                                            source.res_pix**2, 1e-30,None))
        conv_source = smooth(source_phot, offset_res_source)
       
        #resample being more carful to conserve flux here, esp. for undersampled features.
        #assumes uniform distribution of flux across pixel.
        accum_source = np.cumsum(np.r_[conv_source[:1], conv_source])*source.red_step #photons/s/m^2
        source_bin_waves = np.r_[source_wave-source.red_step/2., source_wave[-1:]+source.red_step/2.]
        out_bin_waves = np.r_[self.inst_wavelength-self.step/2., self.inst_wavelength[-1:]+self.step/2.]
        interp_source = np.interp(out_bin_waves, source_bin_waves, accum_source)
        source_resampled = np.diff(interp_source)/self.step
        
        #source_resampled = np.interp(self.inst_wavelength, source_wave, conv_source)
        
        return source_resampled #photons/s/m^2/um


    def calc_sn(self, source, sky=None, lgs=None, dit=3600., 
                ndit=None, sn=None, seeing=1., binning=1, ref_wl=0.7, 
                strehl=None):
        """ 
        Estimates the SN given dit and ndit, or ndit given sn and dit.
        """
        
        #generate source
        source_resampled = self.make_source_spectrum(source)
 
        #if a sky object is also supplied, convolve it to match the instrument properties
        if sky is not None:
            sky_emm_resampled, sky_trans_resampled = self.make_sky_spectrum(sky)
        else:
            sky_trans_resampled = np.ones(len(self.inst_wavelength))
            sky_emm_resampled = np.zeros(len(self.inst_wavelength))

        #if an LGS object is supplied generate the appropriate spectrum to be included
        if lgs is not None:
            lgs_resampled = self.make_source_spectrum(lgs)
        else:
            lgs_resampled = np.zeros(len(self.inst_wavelength))

        #store transmission spectrum
        self.sky_trans = np.copy(sky_trans_resampled)
        
        #estimate the ensquared energy and pixel area
        self.obs_ee, self.obs_area = self._ee(seeing, binning=binning, strehl=strehl)
        
        #total source spectrum
        if source.norm_sb:
            self.cfact = sky_trans_resampled*dit*self.total_throughput*self.step*\
                         self.telescope.area*self.pix_scale**2 * self.obs_area
        else:
            #get ensquared energy and area in pixels
            self.cfact = sky_trans_resampled*dit*self.total_throughput*self.step*\
                         self.telescope.area*self.obs_ee

        self.source_obs = np.copy(source_resampled)*self.cfact

        #sky is always done correctly-ish.
        self.sky_obs = np.copy(sky_emm_resampled)*dit*self.total_throughput*\
                  self.step*self.telescope.area*self.pix_scale**2 * self.obs_area #total area
           
        #lgs gets scaled in the same way as sky
        self.lgs_obs = np.copy(lgs_resampled)*dit*self.total_throughput*\
                  self.step*self.telescope.area*self.pix_scale**2 * self.obs_area

        #total noise calculation. Includes term for sampling of spatial elements by detector (factor 1.05)
        self.noise = self.source_obs + self.sky_obs + self.lgs_obs + (self.obs_area*1.05**2)*(self.detector.rn**2 + self.detector.dark*dit) #per dit
        
        if sn is not None and ndit is None: #provided S/N, work out ndit to reach S/N 5
            ndit = np.sqrt(self.noise)*sn/self.source_obs
            ndit_i = int(np.ceil(np.interp(ref_wl, self.inst_wavelength, ndit)))
            print("NDIT={2} to reach S/N={0} with DIT={1}s at {3:6.4f}um".format(sn, dit, ndit_i, ref_wl))
        elif sn is None and ndit is not None:
            sn = np.sqrt(ndit)*self.source_obs / np.sqrt(self.noise)
            sn_i = np.interp(ref_wl, self.inst_wavelength, sn) 
            print("S/N={0:4.2f} at {3:6.4f}um with NDIT={1} and DIT={2}".format(sn_i, ndit, dit, ref_wl))

        sn = np.sqrt(ndit)*self.source_obs / np.sqrt(self.noise)
        return self.inst_wavelength, sn


    def get_mag_limit(self, sn=5, sky=None, lgs=None, dit=3600., ndit=1, 
                        seeing=1., binning=1, ref_wl=0.7, 
                        strehl=None, norm='point'):
        """
        Estimate magnitude limits given observing parameters.
        """

        #if a sky object is also supplied, convolve it to match the instrument properties
        if sky is not None:
            sky_emm_resampled, sky_trans_resampled = self.make_sky_spectrum(sky)
        else:
            sky_trans_resampled = np.ones(len(self.inst_wavelength))
            sky_emm_resampled = np.zeros(len(self.inst_wavelength))

        #if an LGS object is supplied generate the appropriate spectrum to be included
        if lgs is not None:
            lgs_resampled = self.make_source_spectrum(lgs)
        else:
            lgs_resampled = np.zeros(len(self.inst_wavelength))

        #store transmission spectrum
        self.sky_trans = np.copy(sky_trans_resampled)
       
        #estimate the ensquared energy and pixel area
        self.obs_ee, self.obs_area = self._ee(seeing, binning=binning, strehl=strehl)
        
        #cfact converts back and forth between counts and physical units
        if norm == 'point':
            self.cfact = sky_trans_resampled*dit*self.total_throughput*self.step*\
                         self.telescope.area*self.obs_ee
        else:
            self.cfact = sky_trans_resampled*dit*self.total_throughput*self.step*\
                         self.telescope.area*self.pix_scale**2 * self.obs_area

        #the sky is basically always okay-ish
        self.sky_obs = np.copy(sky_emm_resampled)*dit*self.total_throughput*\
                  self.step*self.telescope.area*self.pix_scale**2 * self.obs_area #total area
            
        #lgs gets treated the same as the sky (sB)
        self.lgs_obs = np.copy(lgs_resampled)*dit*self.total_throughput*\
                  self.step*self.telescope.area*self.pix_scale**2 * self.obs_area #total area
        
        #quadratic terms for solution. noise term includes sampling on detector.
        a = ndit / sn**2
        c = self.sky_obs + self.lgs_obs + (self.obs_area*1.05**2)*(self.detector.rn**2 + self.detector.dark*dit)

        source_obs = 0.5*(1. + np.sqrt(1. + 4*a*c)) * sn**2 / ndit

        return -2.5*np.log10(source_obs * self.inst_wavelength * 6.626196e-27 / 100**2 / self.cfact)-48.6


    def observe(self, source, sky=None, lgs=None, dit=3600., 
                ndit=1, seeing=1., binning=1, ref_wl=0.7, 
                strehl=None, combine='mean'):
        """
        Generate a simulated spectrum and corresponding noise given dit and ndit.
        """

        #generate source
        source_resampled = self.make_source_spectrum(source)

        if source.type == 'lamp': #separate handling for lamp-type sources
            #plate scale in "/mm at the PFR output
            plate_scale = 0.737
            
            #source_resmapled has units of ph/s/m^2/micron at the instrument focal plane 
            #need to do some conversion to get this into the frame of the spectrograph
            #output is phot/pixel at the detector.
            self.source_obs = np.copy(source_resampled)*self.total_throughput*self.step*dit*\
                                self.pix_scale**2 / (plate_scale*1000)**2 / self.telescope_throughput / 1.05**2 #spatial->detector mapping?

            self.noise = self.source_obs + self.detector.rn**2 + self.detector.dark*dit

            # generate noisy realisations of the data
            rng = np.random.default_rng()
            spec_all = self.source_obs[np.newaxis, :] + rng.normal(loc=0, scale=np.tile(np.sqrt(self.noise), (ndit,1))) 
           
            spec_out = np.mean(spec_all, axis=0)

            return spec_out, self.source_obs

        else:

            #if a sky object is also supplied, convolve it to match the instrument properties
            if sky is not None:
                sky_emm_resampled, sky_trans_resampled = self.make_sky_spectrum(sky)
            else:
                sky_trans_resampled = np.ones(len(self.inst_wavelength))
                sky_emm_resampled = np.zeros(len(self.inst_wavelength))
    
            #if an LGS object is supplied generate the appropriate spectrum to be included
            if lgs is not None:
                lgs_resampled = self.make_source_spectrum(lgs)
            else:
                lgs_resampled = np.zeros(len(self.inst_wavelength))
    
    
            #store transmission spectrum
            self.sky_trans = np.copy(sky_trans_resampled)
            
            #estimate the ensquared energy and pixel area
            self.obs_ee, self.obs_area = self._ee(seeing, binning=binning, strehl=strehl)
            
            #total source spectrum
            if source.norm_sb:
                self.cfact = sky_trans_resampled*dit*self.total_throughput*self.step*\
                             self.telescope.area*self.pix_scale**2 * self.obs_area
            else:
                #get ensquared energy and area in pixels
                self.cfact = sky_trans_resampled*dit*self.total_throughput*self.step*\
                             self.telescope.area*self.obs_ee
    
            self.source_obs = np.copy(source_resampled)*self.cfact
    
            #sky is always done correctly-ish.
            self.sky_obs = np.copy(sky_emm_resampled)*dit*self.total_throughput*\
                      self.step*self.telescope.area*self.pix_scale**2 * self.obs_area #total area
    
            #sky is always done correctly-ish.
            self.lgs_obs = np.copy(lgs_resampled)*dit*self.total_throughput*\
                      self.step*self.telescope.area*self.pix_scale**2 * self.obs_area #total area
            
            self.noise = self.source_obs + self.sky_obs + self.lgs_obs + (self.obs_area*1.05**2)*(self.detector.rn**2 + self.detector.dark*dit) #per dit
    
            # generate noisy realisations of the data
            rng = np.random.default_rng()
            spec_all = self.source_obs[np.newaxis, :] + rng.normal(loc=0, scale=np.tile(np.sqrt(self.noise), (ndit,1))) 
           
            spec_out = np.mean(spec_all, axis=0)
            spec_out[self.notch] = 0.
    
    
            return (spec_out * 6.626196e-27 * 2.998e14 / self.inst_wavelength / self.cfact / 100**2 / 1e4, 
                    self.source_obs * 6.626196e-27 * 2.998e14 / self.inst_wavelength / self.cfact / 100**2 / 1e4)



class MAVIS_IFS(IFSInstrument):
    """
    A MAVIS-like integral field spectrograph.

    Assumes that the general properties of the MAVIS spectrograph
    are MUSE-like, but with an added throughput hit from the AO system
    and more elaborate PSF model.
    """

    def __init__(self, mode=None, pix_scale=0.025, jitter=5, live_fraction=0.95, 
                 detector=None, telescope=None, inst_wavelength=None, turbulence_cat="50%-plus-spec",
                 throughput_model="2025-03-06", aom_model="2025-03-14"):
        
        #check for reasonable jitter values
        if jitter not in [5,10,20,30,40]:
            raise ValueError('Input jitter must be one of 5, 10, 20, 30, or 40 (mas)')

        if mode not in ['LR-blue','LR-red','HR-blue','HR-red']:
            raise ValueError('Invalid grating setup.  Must be one of LR-blue, LR-red, HR-blue, or HR-red.')

        if pix_scale not in [0.025,0.050]:
            raise ValueError('Invalid pixel scale. Must be one 0.025 or 0.05 arcsec')

        #initialize the model base
        IFSInstrument.__init__(self)

        #set the pixel scale
        self.pix_scale = pix_scale

        #initialize the provided detector object
        if detector is None:
            self.detector = CCD290()
        else:
            self.detector = detector

        #detector parameters 
        self.live_pix_det = self.detector.npix_det * live_fraction

        #grating parameters
        self.grating_dict = {
                'LR-blue': (3700., 3450.),
                'LR-red': (5150., 3450.),
                'HR-blue': (4250., 12800.),
                'HR-red': (6300., 9600.),
                }
        self.wmin, self.wmax = 3700./1e4, 9400./1e4

        #pull correct grating parameters
        grating_wmin, grating_rmin = self.grating_dict[mode]

        #generate full wavelength and resolution arrays
        self.dlam = grating_wmin / grating_rmin / 1e4
        self.step = self.dlam / 2.1

        npix_wave = min(self.live_pix_det, int(np.floor(grating_wmin/self.step/1e4))) #avoid coverage over an octave

        if inst_wavelength is not None:
            self.inst_wavelength = inst_wavelength
            self.wmin_eff, self.wmax_eff = self.inst_wavelength.min(), self.inst_wavelength.max()
            self.res_power_interp = np.ones_like(self.inst_wavelength)*3500.
            self.res_pix = self.inst_wavelength / self.res_power_interp / self.step / 2.355
        else:   
            self.inst_wavelength = np.arange(npix_wave)*self.step + grating_wmin / 1e4
            self.wmin_eff, self.wmax_eff = self.inst_wavelength[0], min(self.inst_wavelength[-1], self.wmax)
            self.res_power_interp = self.inst_wavelength / 2.1 / self.step
            self.res_pix = self.inst_wavelength / self.res_power_interp / self.step / 2.355

        #get path for bundled package files
        bfile_dir = os.path.join(os.path.dirname(sys.modules['mavisetc'].__file__), 'data')
        
        #get detector QE
        self.qe = self.detector.qe_interp(self.inst_wavelength)

        #mode- and scale-dependant throughput from optical model        
        if pix_scale == 0.025:
            tpt_dict = {
                'LR-blue': '25LRB',
                'LR-red': '25LRR',
                'HR-blue': '25HRB',
                'HR-red': '25HRR',
                }
        if pix_scale == 0.050:
            tpt_dict = {
                'LR-blue': '50LRB',
                'LR-red': '50LRR',
                'HR-blue': '50HRB',
                'HR-red': '50HRR',
                }

        #build the throughput file
        tpt_file = 'MAVIS_throughput_spec_{0}.csv'.format(throughput_model)
        if not os.path.exists(os.path.join(bfile_dir, 'mavis', tpt_file)):
            raise ValueError('Throughput model {0} not found'.format(tpt_file)) 

        data = asc.read(os.path.join(bfile_dir, 'mavis', tpt_file), encoding='utf-8-sig')
        twave = np.array(data['Wavelength'])/1e3
        ttpt = np.array(data[tpt_dict[mode]])
        self.instrument_throughput = interp1d(twave[ttpt > 0], ttpt[ttpt > 0], fill_value='extrapolate')(self.inst_wavelength)



        #initialize the telescope
        if telescope is None:
            self.telescope = VLT()
        else:
            self.telescope = telescope()

        #if the instrument throughput curves are already included, this should be set to 1
        self.telescope_throughput = np.interp(self.inst_wavelength, self.telescope.telescope_wave, self.telescope.telescope_eff)

        #compute the combined throughput
        self.total_throughput = self.telescope_throughput*\
                          self.qe*\
                          self.instrument_throughput*0.75*0.55 # re-scaled to account for additional losses not currently dealt with
                                                                     # in the throughput model 

        #patch in low throughput at the notch
        self.notch = (self.inst_wavelength > 0.580) & (self.inst_wavelength < 0.597)

        #AOM throughput estimate
        aom_file = 'mavis_AOM_throughput_{0}_spec.csv'.format(aom_model)
        if not os.path.exists(os.path.join(bfile_dir, 'mavis', aom_file)):
            raise ValueError('AOM model {0} not found'.format(aom_file)) 
        
        data = np.array(asc.read(os.path.join(bfile_dir, 'mavis', aom_file)))
        twave = np.array(data['col1'])
        ttpt = np.array(data['col2'])

        self.ao_throughput = interp1d(np.array(twave), np.array(ttpt), bounds_error=False,
                                    fill_value='extrapolate')(self.inst_wavelength).clip(0,1)


        #include notch throughput as part of the AO system
        data = np.array(asc.read(os.path.join(bfile_dir, 'mavis/notch_throughput.csv')))
        twave = np.array(data['col1'])/1000.
        ttpt = np.array(data['col2'])/100.

        #self.notch_throughput = interp1d(np.array(twave), np.array(ttpt), bounds_error=False,
        #                            fill_value='extrapolate')(self.inst_wavelength).clip(0,1)
        self.notch_throughput = np.ones(len(self.inst_wavelength)) #flat because it's folded into the AOM budget

        self.ao_throughput *= self.notch_throughput

        #fold in AOM+notch to the throughput budget
        self.total_throughput *= self.ao_throughput

        #loading in new PSF library
        turbulence_dict = {'10%': 'PSF_PC10_2024-06-27.fits',
                           '25%': 'PSF_PC25_2024-06-27.fits',
                           '50%': 'PSF_PC50_2024-06-27.fits',
                           '75%': 'PSF_PC75_2024-06-27.fits',
                           '90%': 'PSF_PC90_2024-06-27.fits',
                           'TLR': 'PSF_TLRatmo_2024-08-28.fits',
                           '50%-plus-spec': 'PSF_PC50_spec-{0}mas_2025-04-03.fits'.format(int(pix_scale*1e3))
                           }
            
        ee_model = os.path.join(bfile_dir, 'mavis/{0}'.format(turbulence_dict[turbulence_cat]))
        self._ee_profile_wave = fits.getdata(ee_model, ext=0)/1e3
        psf_rad = fits.getdata(ee_model, ext=1)/self.pix_scale
        psf_ee = fits.getdata(ee_model, ext=2)

        #store the interpolator object
        self._ee_profile_interp = interp1d(psf_rad, psf_ee, axis=0)

        #assign EE generator
        self._ee = self._EE_lookup


    def _EE_lookup(self, seeing, binning=1, **kwargs):
        """
        A quick and dirty interpolation function to generate ensquared 
        energy within a given aperture using the pre-computed MAVIS PSF model..
        """

        #given input binning/radius, generate EE array at each modelled wavelength
        ee_out = self._ee_profile_interp(binning/2)
        
        #now linearly interpolate to every wavelength
        ee_interped = interp1d(self._ee_profile_wave, ee_out, fill_value='extrapolate')(self.inst_wavelength)

        return ee_interped, binning**2


