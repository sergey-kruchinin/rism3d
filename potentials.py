import numpy as np


def make_inverse_long_coulomb_potential(grid, charge1, charge2, 
                                        smear=1, grid_zero=1e-6):
    """Calculate Fourier transformed long range part of Coulomb
    potential. Splitting of the potential on long and short parts is
    performed by the error function. Units of returned value can be 
    changed by multiplying charges on corresponding coefficient.

    Parameters:
        grid : ndarray with grid points.
        charge1 : first electric charge.
        charge2 : second electric charge.
        smear : splitting parameter for error function. Default 1.
        grid_zero : grid values lesser than this parameter will be 
                    set equal to value grid_zero. It will helps to
                    avoid singularities at grid values closed to 0.
    Returns:
        Array [ndarray] of potential values. 
    """
    zero_points = grid < grid_zero
    grid[zero_points] = grid_zero
    v = (charge1 * charge2 / (np.pi * grid**2) 
         * np.exp(-np.pi**2 * grid**2 / smear**2))
    return v 
