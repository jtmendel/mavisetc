import numpy as np
    
def interp(xout, xin, yin):
    """Applies `numpy.interp` to the last dimension of `yin`

    This is also taken directly from pPXF.
    """    

    yout = [np.interp(xout, xin, y) for y in yin.reshape(-1, xin.size)]
    return np.reshape(yout, (*yin.shape[:-1], -1))
        

def smooth(x, y, sig_x, xout=None, oversample=1):
    """    
    Fast and accurate convolution with a Gaussian of variable width.

    This function performs an accurate Fourier convolution of a vector, or the
    columns of an array, with a Gaussian kernel that has a varying or constant
    standard deviation (sigma) per pixel. The convolution is done using fast
    Fourier transform (FFT) and the analytic expression of the Fourier
    transform of the Gaussian function, like in the pPXF method. This allows
    for an accurate convolution even when the Gaussian is severely
    undersampled.

    This function is recommended over standard convolution even when dealing
    with a constant Gaussian width, due to its more accurate handling of
    undersampling issues.

    This function implements Algorithm 1 in `Cappellari (2023)
    <https://ui.adsabs.harvard.edu/abs/2023MNRAS.526.3273C>`_

    This implementation was taken directly from the pPXF package.

    Input Parameters
    ----------------

    x : array_like
        Coordinate of every pixel in `y`.
    y : array_like
        Input vector or array of column-spectra.
    sig_x : float or array_like
        Gaussian sigma of every pixel in units of `x`.
        If sigma is constant, `sig_x` can be a scalar. 
        In this case, `x` must be uniformly sampled.
    oversample : float, optional
        Oversampling factor before convolution (default: 1).
    xout : array_like, optional
        Output `x` coordinate used to compute the convolved `y`.

    Output Parameters
    -----------------

    yout : array_like
        Convolved vector or columns of the array `y`.

    """
    assert len(x) == len(y), "`x` and `y` must have the same length"

    if np.isscalar(sig_x):
        dx = np.diff(x)
        assert np.all(np.isclose(dx[0], dx)), "`x` must be uniformly spaced, when `sig_x` is a scalar"
        n = len(x)
        sig_max = sig_x*(n - 1)/(x[-1] - x[0])
        y_new = y.T
    else:
        assert len(x) == len(sig_x), "`x` and `sig_x` must have the same length"
        # Stretches spectrum to have equal sigma in the new coordinate
        sig = sig_x/np.gradient(x)
        sig = sig.clip(0.1)   # Clip to >=0.1 pixels
        sig_max = np.max(sig)*oversample
        xs = np.cumsum(sig_max/sig)
        n = int(np.ceil(xs[-1] - xs[0]))
        x_new = np.linspace(xs[0], xs[-1], n)
        y_new = interp(x_new, xs, y.T)

    # Convolve spectrum with a Gaussian using analytic FT like pPXF
    npad = 2**int(np.ceil(np.log2(n)))
    ft = np.fft.rfft(y_new, npad)
    w = np.linspace(0, np.pi*sig_max, ft.shape[-1])
    ft_gau = np.exp(-0.5*w**2)
    yout = np.fft.irfft(ft*ft_gau, npad).T[:n]

    if not np.isscalar(sig_x):
        if xout is not None:
            xs = np.interp(xout, x, xs)  # xs is 1-dim
        yout = interp(xs, x_new, yout.T).T

    return yout
