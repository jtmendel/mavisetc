import numpy as np
cimport numpy as np
import cython
from libc.math cimport pow, sqrt, exp, M_PI, floor

DTYPE = float
ctypedef np.float_t DTYPE_t
ctypedef np.int_t DTYPE_i

__all__ = ["smooth"]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def smooth(np.ndarray[DTYPE_t, ndim=1] flux, np.ndarray[DTYPE_t, ndim=1] sigma):

    cdef Py_ssize_t N_elem = flux.shape[0]
    cdef Py_ssize_t pix, npix

    cdef np.ndarray[DTYPE_t, ndim=1] out_spec = np.zeros(N_elem, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] norm_spec = np.ones(N_elem, dtype=DTYPE) 
    cdef np.ndarray[DTYPE_t, ndim=1] ttpix = np.zeros(N_elem, dtype=DTYPE)

    cdef np.ndarray[DTYPE_t, ndim=1]  width = 2.355*6*sigma #sample the distribution out to 3*FWHM
    cdef np.ndarray[DTYPE_i, ndim=1]  spix = np.asarray(np.floor(sigma*6.), dtype=int)

    for pix in range(N_elem):
        if width[pix]/2.355 < 1.0:
            out_spec[pix] = flux[pix]
            ttpix[pix] = norm_spec[pix]
        else:
            if pix+1 < width[pix]:
                for npix in range(0-pix,spix[pix]):
                    out_spec[npix+pix] = out_spec[npix+pix] + flux[pix]*(1./sigma[pix]/sqrt(2*M_PI))*exp(-0.5*pow(npix/sigma[pix],2))
                    ttpix[npix+pix] = ttpix[npix+pix] + norm_spec[pix]*(1./sigma[pix]/sqrt(2*M_PI))*exp(-0.5*pow(npix/sigma[pix],2))
            elif width[pix] <= pix+1 <= N_elem-width[pix]+1:
                for npix in range(-1*spix[pix]+1,spix[pix]):    
                    out_spec[npix+pix] = out_spec[npix+pix] + flux[pix]*(1./sigma[pix]/sqrt(2*M_PI))*exp(-0.5*pow(npix/sigma[pix],2))
                    ttpix[npix+pix] = ttpix[npix+pix] + norm_spec[pix]*(1./sigma[pix]/sqrt(2*M_PI))*exp(-0.5*pow(npix/sigma[pix],2))
            else:     
                for npix in range(-1*spix[pix]+1,N_elem-pix):
                    out_spec[npix+pix] = out_spec[npix+pix] + flux[pix]*(1./sigma[pix]/sqrt(2*M_PI))*exp(-0.5*pow(npix/sigma[pix],2))
                    ttpix[npix+pix] = ttpix[npix+pix] + norm_spec[pix]*(1./sigma[pix]/sqrt(2*M_PI))*exp(-0.5*pow(npix/sigma[pix],2))


    return out_spec/ttpix


