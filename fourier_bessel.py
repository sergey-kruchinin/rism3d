import numpy as np
from scipy import fftpack
from scipy import interpolate


def inverse_fourier_bessel(data, k_grid):
    """Calculate inverse Fourier-Bessel transform.

    Calculates inverse Fourier-Bessel transform of AmberTools 1drism
    data. Supposes the first grid point is equals zero, 
    so transformation performed on last N-1 points and first point
    is extrapolated. The coefficient of transformation dk / 4 / pi**2 
    was obtained empirically.
    """
    f = np.zeros_like(data)
    f[1:] = fftpack.dst((data * k_grid)[1:], type=1)
    k_delta = k_grid[1] - k_grid[0]
    npoints = len(k_grid)
    r_delta = np.pi / npoints / k_delta
    r_grid = np.arange(npoints) * r_delta
    f[1:] /= r_grid[1:]
    f_interp = interpolate.interp1d(r_grid[1:], 
                                    f[1:], 
                                    kind="cubic", 
                                    fill_value="extrapolate")
    f[0] = f_interp(r_grid[0])
    f *= k_delta / 4 / np.pi**2
    return f
