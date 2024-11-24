import numpy as np
from scipy import special


def get_inverse_long_coulomb(grid, charge1, charge2, smear=1, grid_zero=1e-6):
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


def get_lj(solute, solvent, box):
    v = 0
    for site in solute.sites:
        d = box.get_distances(origin=site.coordinates)
        d[d < 1e-6] = 1e-6
        d = 1.0 / d
        r_min = site.rmin + solvent.rmin
        frac = np.tensordot(r_min, d, axes=0)**6
        eps = np.expand_dims(np.sqrt(site.epsilon * solvent.epsilon),
                             axis=(1, 2, 3))
        v += eps * (frac**2 - 2 * frac)
    return v


def get_short_coulomb(solute, solvent, box, smear):
    v = 0
    for site in solute.sites:
        d = box.get_distances(origin=site.coordinates)
        d[d < 1e-6] = 1e-6
        v += site.charge * special.erfc(d * smear) / d
    v = np.tensordot(solvent.charge, v, axes=0) 
    return v


def get_long_coulomb(solute, solvent, box, smear):
    """Calculate long coulomb potential."""
    v = 0
    for site in solute.sites:
        d = box.get_distances(origin=site.coordinates)
        d[d < 1e-6] = 1e-6
        v += site.charge * special.erf(d * smear) / d
    v = np.tensordot(solvent.charge, v, axes=0)
    return v

