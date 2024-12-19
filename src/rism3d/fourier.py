import numpy as np
from scipy import fftpack
from scipy import interpolate


def calculate_inverse_fourier_bessel(data, r_delta, k_delta):
    """Calculate inverse Fourier-Bessel transform. 
    The input functions should be tabulated on uniformly spaced grid. 
    In case of array of functions transformation is performed over 
    last dimension of data array. 

    Parameters:
        data : array of input values.
        r_delta : the grid spacing in the direct space.
        k_delta : the grid spacing in the inverse space.
    Returns:
        array of values of transformed functions.
    Raises:
        ValueError : If any value of input data 
                     is non-finite (inf's or NaN's) 
    """
    if np.any(~np.isfinite(data)):
        raise ValueError("Input data consists infinities or NaN's")
    index = np.arange(data.shape[-1])
    f = np.zeros_like(data)
    f[..., 0] = 4 * np.pi * k_delta**3 * np.sum(data * index**2, axis=-1)
    f[..., 1:] = (k_delta**2 / (index[1:] * r_delta) 
             * fftpack.dst(data[..., 1:] * index[1:], type=1))
    return f
