from __future__ import print_function
from astropy.io import fits, ascii as asc
from astropy.convolution import Gaussian2DKernel
from scipy.interpolate import interp1d

import numpy as np
import os
import glob
import sys
import fsps

from ..utils.smoothing import smooth

#import some bits for the instruments
from ..detectors import CCD250
from ..telescopes import VLT

__all__ = ["MAVIS_Imager"]


class ImagingInstrument:
    """
    Generic instrument class with some generic instrument routines.  
    """
    def __init__(self):
        self.small_num = 1e-70
        self.source_obs = None
        self.sky_obs = None
   
        self.transmission = None
        self.wavelength = None
        self.pivot = None

    def _EE_gaussian(self, seeing, binning=1, **kwargs):
        sigma_pix = seeing / self.pix_scale / 2.355 #relies on the v-band seeing here

        wave_temp = np.linspace(self.inst_wavelength.min(), self.inst_wavelength.max(), 50)
        sig_temp = sigma_pix * (wave_temp/.500)**(-1./5.)
        ee_temp = np.zeros_like(wave_temp)
        for ii, tsig in enumerate(sig_temp):
            ee_temp[ii] = Gaussian2DKernel(tsig, x_size=binning, y_size=binning,
                                            mode='oversample', factor=100).array.sum()

        ee_array = interp1d(wave_temp, ee_temp, kind='quadratic')(self.inst_wavelength)

        return ee_array, binning**2    

    def _set_filter(self, filt):
        #fetch filter transmission curve from FSPS
        #this sets the wavelength grid, so no rebinning needed

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
        trans_interp = np.asarray(np.interp(self.inst_wavelength, fwl/1e4, 
                                  ftrans, left=0., right=0.), dtype=np.float)

        #normalize transmission
        #ttrans = np.trapz(np.copy(trans_interp)/self.inst_wavelength, self.inst_wavelength)
        ttrans = np.trapz(np.copy(trans_interp), self.inst_wavelength)
        if ttrans < self.small_num: ttrans = 1.
        ntrans = np.maximum(trans_interp / ttrans, 0.0)
        
        if filter_num in mips_dict:
            td = np.trapz(((self.inst_wavelength/mips_dict[filter_num])**(-2.))*ntrans/self.inst_wavelength, self.inst_wavelength)
            ntrans = ntrans/max(1e-70,td)

        if filter_num in spitzer_dict:
            td = np.trapz(((self.inst_wavelength/spitzer_dict[filter_num])**(-1.0))*ntrans/self.inst_wavelength, self.inst_wavelength)
            ntrans = ntrans/max(1e-70,td)

        #stupid, but re-normalize to peak of 1 (since all other throughput terms 
        #are included in the instrument throughput

        self.trans_norm = np.copy(ntrans)/ntrans.max()
        self.transmission = ntrans
        self.pivot = fobj.lambda_eff
        return 

    def _patch_nan(self, signal):
        nans = np.isnan(signal)
        interpolator = interp1d(self.inst_wavelength[~nans], signal[~nans], bounds_error='extrapolate')
        signal[nans] = interpolator(self.inst_wavelength[nans])
        return signal


    def make_sky_spectrum(self, sky, source, source_wave):
        sky_wave, sky_emm, sky_trans = sky()
       
        match_res_sky = np.interp(sky_wave, source_wave, source.res_pix,
                                 left=source.res_pix[0], right=source.res_pix[-1])
        offset_res_sky = np.sqrt(np.clip((match_res_sky*source.step/sky.step)**2 - 
                                         sky.res_pix**2, 1e-10, None))
        conv_emm = smooth(sky_emm, offset_res_sky)
        conv_trans = smooth(sky_trans, offset_res_sky)
                    
        #resample onto output grid
        sky_emm_resampled = np.interp(self.inst_wavelength, sky_wave, conv_emm)
        sky_trans_resampled = np.clip(np.interp(self.inst_wavelength, 
                                                     sky_wave, conv_trans),0,1)

        return sky_emm_resampled, sky_trans_resampled


    def calc_sn(self, source, sky=None, dit=3600., 
                ndit=None, sn=None, seeing=1., binning=1, band='v', strehl=None):
    
        #generate source spectrum
        source_wave, source_phot = source()
        
        #resample onto outputpixel grid
        source_resampled = np.interp(self.inst_wavelength, source_wave, source_phot)
 
        #if a sky object is also supplied, convolve it to match the instrument properties
        if sky is not None:
            sky_emm_resampled, sky_trans_resampled = self.make_sky_spectrum(sky, source, source_wave)
        else:
            sky_trans_resampled = np.ones(len(self.inst_wavelength))
            sky_emm_resampled = np.zeros(len(self.inst_wavelength))

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
            self.cfact = sky_trans_resampled*dit*self.total_throughput*\
                         self.step*self.telescope.area*self.obs_ee

        source_obs = np.copy(source_resampled)*self.cfact #photons

        #sky is always done correctly-ish.
        sky_obs = np.copy(sky_emm_resampled)*dit*self.total_throughput*self.step*\
                  self.telescope.area*self.pix_scale**2 * self.obs_area #total area, photons#/um


        ##set filter
        for iband in band:
            self._set_filter(iband)
         
            #integrate transmission for total counts
            self.source_obs = self._patch_nan(source_obs)*self.trans_norm
            self.sky_obs = self._patch_nan(sky_obs)*self.trans_norm
        
            #total noise calculation
            self.dark_noise = self.obs_area*self.detector.dark*dit
            self.read_noise = self.obs_area*self.detector.rn**2
            self.sky_noise = np.nansum(self.sky_obs)
            self.obj_noise = np.nansum(self.source_obs)
            self.noise = self.obj_noise + self.sky_noise + self.read_noise + self.dark_noise #per dit
        
            if sn is not None and ndit is None: #provided S/N, work out ndit to reach target S/N
                indit = np.int(np.ceil(np.sqrt(self.noise)*sn/self.source_obs.sum()))
                print("NDIT={2} to reach S/N={0} with DIT={1} in {3}".format(sn, dit, indit, iband))
            elif sn is None and ndit is not None:
                isn = np.sqrt(ndit)*self.source_obs.sum() / np.sqrt(self.noise)
                print("S/N={0:4.2f} at with NDIT={1} and DIT={2} in {3}".format(isn, ndit, dit, iband))

        return 


    def get_mag_limit(self, sn=None, sky=None, dit=3600., 
                      ndit=None, seeing=1., binning=1, band='v', strehl=None,
                      norm='point'):
    

        #if a sky object is also supplied, convolve it to match the instrument properties
        if sky is not None:
            sky_wave, sky_emm, sky_trans = sky()
            
            #resample onto output grid
            sky_emm_resampled = np.interp(self.inst_wavelength, sky_wave, sky_emm)
            sky_trans_resampled = np.clip(np.interp(self.inst_wavelength, 
                                                         sky_wave, sky_trans),0,1)
        else:
            sky_trans_resampled = np.ones(len(self.inst_wavelength))
            sky_emm_resampled = np.zeros(len(self.inst_wavelength))

        #store transmission spectrum
        self.sky_trans = np.copy(sky_trans_resampled)
        
        #estimate the ensquared energy and pixel area
        self.obs_ee, self.obs_area = self._ee(seeing, binning=binning, strehl=strehl)

        #total source spectrum
        if norm == 'point':
            #get ensquared energy and area in pixels
            self.cfact = sky_trans_resampled*dit*self.total_throughput*\
                         self.step*self.telescope.area*self.obs_ee
        else:
            self.cfact = sky_trans_resampled*dit*self.total_throughput*self.step*\
                         self.telescope.area*self.pix_scale**2 * self.obs_area

        #sky is always done correctly-ish.
        sky_obs = np.copy(sky_emm_resampled)*dit*self.total_throughput*self.step*\
                  self.telescope.area*self.pix_scale**2 * self.obs_area #total area, photons#/um

        ##set filter
        store_limit = []
        store_pivot = []
        for iband in band:
            self._set_filter(iband)
         
            #integrate transmission for total counts
            self.sky_obs = self._patch_nan(sky_obs)*self.trans_norm
        
            #total noise calculation
            self.dark_noise = self.obs_area*self.detector.dark*dit
            self.read_noise = self.obs_area*self.detector.rn**2
            self.sky_noise = np.nansum(self.sky_obs)
            self.noise = self.sky_noise + self.read_noise + self.dark_noise #per dit
       
            #quadratic terms for solution
            a = ndit / sn**2
            c = self.noise

            source_obs = 0.5*(1. + np.sqrt(1. + 4*a*c)) * sn**2 / ndit #total counts

            #print(source_obs, self.pivot)
            store_limit.append(-2.5*np.log10(source_obs * self.pivot * 6.626196e-27 / 100**2 / np.nansum(self.cfact*self.trans_norm) / 1e4)-48.6)
            store_pivot.append(self.pivot/1e4)

        return store_pivot, store_limit



class MAVIS_Imager(ImagingInstrument):
    """
    A MAVIS-like instrument.

    Assumes that the general properties of the MAVIS spectrograph
    are MUSE-like, but with an added throughput hit from the AO system
    and more elaborate PSF model.
    """

    def __init__(self, pix_scale=0.00736, jitter=5, detector=None, telescope=None):
        #check for reasonable jitter values
        if jitter not in [5,10,20,30,40]:
            raise ValueError('Input jitter must be one of 5, 10, 20, 30, or 40 (mas)')

        #initialize the model base
        ImagingInstrument.__init__(self)

        #set the pixel scale
        self.pix_scale = pix_scale
       
        #wavelength business
        self.step = 1. / 1e4 #microns
        self.wmin, self.wmax = 3700./1e4, 10100./1e4
        self.inst_wavelength = np.arange(self.wmin, self.wmax, self.step)

        #initialize the provided detector object
        if detector is None:
            self.detector = CCD250()
        else:
            self.detector = detector()

        #get detector QE
        self.qe = self.detector.qe_interp(self.inst_wavelength)
        
        #initialize the telescope
        if telescope is None:
            self.telescope = VLT()
        else:
            self.telescope = telescope()

        #if the instrument throughput curves are already included, this should be set to 1
        self.telescope_throughput = np.interp(self.inst_wavelength, self.telescope.telescope_wave, self.telescope.telescope_eff)

        #compute the combined throughput - including filter transmission
        self.total_throughput = self.telescope_throughput * self.qe * 0.98**2

        #patch in low throughput at the notch
        self.notch = (self.inst_wavelength > 0.580) & (self.inst_wavelength < 0.597)

        #get path for bundled package files
        bfile_dir = os.path.join(os.path.dirname(sys.modules['mavisetc'].__file__), 'data')

        #fold in AOM throughput
        data = np.array(asc.read(os.path.join(bfile_dir, 'mavis/mavis_AOM_throughput.csv')))
        twave = np.array(data['col1'])
        ttpt = np.array(data['col2'])

        self.ao_throughput = np.interp(self.inst_wavelength, np.array(twave), np.array(ttpt),
                                left=ttpt[0], right=ttpt[-1])
        self.total_throughput *= self.ao_throughput

        #pre-load EE profiles
        ee_files = glob.glob(os.path.join(bfile_dir, 'mavis/PSF_{0}mas*EEProfile.dat'.format(jitter)))
        wave = []
        ee_interp = []
        for ii, ee_file in enumerate(ee_files):
            twave = os.path.split(ee_file)[1].split('_')[2][:-2]
            wave.append(float(twave)/1e3)
            with open(ee_file, 'r') as file:
                tr, tee = [], []
                for line in file:
                    temp = line.strip().split(',')
                    tr.append(float(temp[0])/self.pix_scale) #in pixels
                    tee.append(float(temp[1]))
                ee_interp.append(interp1d(tr, tee, bounds_error=False, fill_value='extrapolate'))
        self._ee_profile_wave = np.array(wave)
        self._ee_profile_interp = ee_interp
        
        self._ee = self._EE_lookup
       

    def _EE_lookup(self, seeing, binning=1, **kwargs):
        """
        A quick and dirty lookup/interpolation function to generate ensquared 
        energy profiles based on simulations of the MAVIS PSF.
        """
        #Seeing isn't relevant for MAVIS at the moment, so it only depends on the binning
        iwave = np.argsort(self._ee_profile_wave)
        
        wave_out = np.zeros(len(iwave), dtype=np.float)
        ee_out = np.zeros(len(iwave), dtype=np.float)
        for ii, idx in enumerate(iwave): #sorted arguments?
            wave_out[ii] = self._ee_profile_wave[idx]
            ee_out[ii] = self._ee_profile_interp[idx](binning/2.)
   
        ee_interped = interp1d(wave_out, ee_out, fill_value='extrapolate')(self.inst_wavelength)
        return ee_interped, binning**2
 
