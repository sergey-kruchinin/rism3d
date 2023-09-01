import copy
import numpy as np
from scipy import fft
from scipy import interpolate
from scipy import special
import itertools
import constants
import mdiis
import exceptions
import potentials
import fourier


class Rism3D:
    def __init__(self, solute, solvent, box, parameters):
        self._solute = solute
        self._solvent = solvent
        self._box = copy.deepcopy(box)
        self._parameters = copy.deepcopy(parameters)
        self._beta = 1 / (constants.k_Boltzmann 
                          * self._parameters["temperature"]) 
        closures = {"hnc": self._use_hnc, 
                    "kh": self._use_kh, 
                    "pse3": self._use_pse3}
        self._use_closure = closures[self._parameters["closure"]]
        solvers = {"picard": self._use_picard_solver, 
                   "mdiis": self._use_mdiis_solver}
        self._use_solver = solvers[self._parameters["solver"]]
        self._r_grid = _get_r_grid(self._box)
        self._r_delta = _get_r_delta(self._r_grid)
        self._k_grid = _get_k_grid(self._box)
        self._chi = _get_susceptibility(self._solvent, self._k_grid)
        self._v_s = _get_short_potential(self._solute, self._solvent, 
                                         self._r_grid, 
                                         self._parameters["smear"], 
                                         self._parameters["dieps"], 
                                         self._beta)
        self._v_l = _get_long_potential(self._solute, self._solvent, 
                                        self._r_grid, 
                                        self._parameters["smear"], 
                                        self._parameters["dieps"], 
                                        self._beta)
        self._theta = _get_renormalized_potential(self._solute, self._solvent, 
                                                  self._r_grid, self._k_grid, 
                                                  self._parameters["smear"], 
                                                  self._beta)
        self._gamma = np.zeros_like(self._v_s)
        
    @property
    def parameters(self):
        return self._parameters

    @property
    def r_delta(self):
        return self._r_delta

    @property
    def beta(self):
        return self._beta

    def solve(self):
        self._use_solver()

    def get_h(self):
        c_s = self._use_closure()
        h = c_s + self._gamma
        return h

    def get_c(self):
        c_s = self._use_closure()
        c = c_s - self._v_l
        return c

    def _use_picard_solver(self):
        mix = self._parameters["mix"]
        gamma_old = self._gamma.copy()
        step = 0
        print("{0:<6s}{1:>18s}".format("step", "accuracy"))
        while True:
            c_s = self._use_closure()
            self._use_oz(c_s)
            self._gamma -= self._theta 
            e = np.max(np.abs(self._gamma - gamma_old))
            self._gamma = mix * self._gamma + (1 - mix) * gamma_old
            step += 1
            gamma_old = self._gamma.copy()
            print("{0:<6d}{1:18.8e}".format(step, e))
            if step >= self._parameters["nsteps"]:
                raise exceptions.Rism3DMaxStepError("The maximum number of steps has been reached", step, e)
            if np.isnan(e) or np.isinf(e):
                raise exceptions.Rism3DConvergenceError("The solution has been diverged", step)
            if e < self._parameters["accuracy"]:
                break
                
    def _use_mdiis_solver(self):
        m = mdiis.MDIIS(self._parameters["mdiis_vectors"],
                        self._parameters["mdiis_mix"],
                        self._parameters["mdiis_max_residue"]
                        )
        gamma_old = self._gamma.copy()
        step = 0
        print("{0:<6s}{1:>18s}{2:>7s}".format("step", "accuracy", "MDIIS"))
        while True:
            c_s = self._use_closure()
            self._use_oz(c_s)
            self._gamma -= self._theta 
            residual = self._gamma - gamma_old
            self._gamma = m.optimize(gamma_old, residual)
            e = np.max(np.abs(residual))
            step += 1
            gamma_old = self._gamma.copy()
            print(f"{step:<6d}{e:18.8e}{m.size():>7d}")
            if step >= self._parameters["nsteps"]:
                raise exceptions.Rism3DMaxStepError("The maximum number of steps has been reached", step, e)
            if np.isnan(e) or np.isinf(e):
                raise exceptions.Rism3DConvergenceError("The solution has been diverged", step)
            if e < self._parameters["accuracy"]:
                break

    def _use_oz(self, c_s):
        c_s_ft = _get_fourier_transform(c_s, self._r_grid, self._k_grid)
        gamma_ft = np.sum(self._chi 
                          * np.expand_dims(c_s_ft, axis=1),
                          axis=0) - c_s_ft
        self._gamma = _get_inverse_fourier_transform(gamma_ft, 
                                                     self._r_grid, 
                                                     self._k_grid)

    def _use_hnc(self):
        c_s = np.exp(-self._v_s + self._gamma) - 1 - self._gamma
        return c_s

    def _use_kh(self):
        c_s = np.zeros_like(self._gamma)
        e = -self._v_s + self._gamma
        c_s[e > 0] = -self._v_s[e > 0]
        c_s[e <= 0] = np.exp(e[e <= 0]) - 1 - self._gamma[e <= 0] 
        return c_s

    def _use_pse3(self):
        c_s = np.zeros_like(self._gamma)
        e = -self._v_s + self._gamma
        c_s[e > 0] = (-self._v_s[e > 0] 
                      + (1.0 / 2.0) * e[e > 0]**2 
                      + (1.0 / 6.0) * e[e > 0]**3)
        c_s[e <= 0] = np.exp(e[e <= 0]) - 1 - self._gamma[e <= 0] 
        return c_s


def _get_susceptibility(solvent, k_grid):
    k_1d = solvent.k_grid
    npoints = len(solvent.k_grid)
    chi_1d = solvent.susceptibility
    k_distances = np.linalg.norm(k_grid, axis=0)
    f = interpolate.interp1d(k_1d, 
                             chi_1d, 
                             kind="cubic", 
                             fill_value="extrapolate")
    chi = f(k_distances)
    return chi 


def _get_lj_potential(solute, solvent, r_grid, beta):
    v = 0
    for site in solute.sites:
        site_position = np.expand_dims(site.coordinates, axis=(1, 2, 3))
        d = np.linalg.norm(r_grid - site_position, axis=0)
        d[d < 1e-6] = 1e-6
        d = 1.0 / d
        r_min = site.rmin + solvent.rmin
        frac = np.tensordot(r_min, d, axes=0)**6
        eps = np.expand_dims(np.sqrt(site.epsilon * solvent.epsilon),
                             axis=(1, 2, 3))
        v += beta * eps * (frac**2 - 2 * frac)
    return v


def _get_short_electrostatic_potential(solute, solvent, r_grid, smear, 
                                       dieps, beta):
    v = 0
    for site in solute.sites:
        site_position = np.expand_dims(site.coordinates, axis=(1, 2, 3))
        d = np.linalg.norm(r_grid - site_position, axis=0)
        d[d < 1e-6] = 1e-6
        v += site.charge * special.erfc(d * smear) / d
    v = np.tensordot(solvent.charge, v, axes=0) * beta / dieps
    return v


def _get_short_potential(solute, solvent, r_grid, smear, dieps, beta):
    v = (_get_lj_potential(solute, solvent, r_grid, beta) 
         + _get_short_electrostatic_potential(solute, solvent, r_grid, smear, 
                                              dieps, beta))
    return v


def _get_long_potential(solute, solvent, r_grid, smear, dieps, beta):
    """Calculate v_long * beta."""
    v = 0
    coef = beta / dieps
    for site in solute.sites:
        site_position = np.expand_dims(site.coordinates, axis=(1, 2, 3))
        d = np.linalg.norm(r_grid - site_position, axis=0)
        d[d < 1e-6] = 1e-6
        v += site.charge * special.erf(d * smear) / d
    v = np.tensordot(solvent.charge, v, axes=0) * beta / dieps
    return v
    

def _get_r_delta(r_grid):
    delta = np.array([r_grid[0, 1, 0, 0] - r_grid[0, 0, 0, 0], 
                      r_grid[1, 0, 1, 0] - r_grid[1, 0, 0, 0],
                      r_grid[2, 0, 0, 1] - r_grid[2, 0, 0, 0]])
    return delta


def _get_fourier_transform(data, r_grid, k_grid):
    dV = np.prod(_get_r_delta(r_grid))
    shift = np.expand_dims(r_grid[:, 0, 0, 0], axis=(1, 2, 3))
    K = np.prod(np.exp(-2j * np.pi * k_grid * shift), axis=0)
    inv = fft.rfftn(data, axes=(-3, -2, -1), workers=-1) * dV * K
    return inv


def _get_inverse_fourier_transform(data, r_grid, k_grid):
    dV = np.prod(_get_r_delta(r_grid))
    shift = np.expand_dims(r_grid[:, 0, 0, 0], axis=(1, 2, 3))
    K = np.prod(np.exp(2j * np.pi * k_grid * shift), axis=0)
    shape = r_grid[0].shape
    inv = fft.irfftn(data * K, s=shape, axes=(-3, -2, -1), workers=-1) / dV
    return inv


def _get_r_grid(box):
    grids = [np.linspace(*i) for i in box]
    r_grid = np.stack(np.meshgrid(*grids, indexing="ij"))
    return r_grid


def _get_k_grid(box):
    points_and_steps = [(i[2], (i[1] - i[0]) / (i[2] - 1)) for i in box]
    grids = [fft.fftfreq(*i) for i in points_and_steps] 
    grids[-1] = fft.rfftfreq(*points_and_steps[-1])
    k_grid = np.stack(np.meshgrid(*grids, indexing="ij"))
    return k_grid

        
def _get_renormalized_potential(solute, solvent, r_grid, k_grid, smear, beta):
    theta_site_ft = np.einsum("i,ijk->jk", solvent.charge, 
                              solvent.susceptibility)
    k_1d = solvent.k_grid
    phi_ft = (potentials.make_inverse_long_coulomb_potential(k_1d, 1, 1, smear)
              * beta) 
    theta_site_ft = theta_site_ft * phi_ft
    k_delta = k_1d[1] - k_1d[0]
    number_of_points = len(k_1d)
    r_delta = 1.0 / (2 * number_of_points * k_delta)
    theta_site = fourier.calculate_inverse_fourier_bessel(theta_site_ft, 
                                                          r_delta, 
                                                          k_delta)
    r_1d = np.arange(number_of_points) * r_delta
    f = interpolate.interp1d(r_1d, 
                             theta_site, 
                             kind="cubic", 
                             fill_value="extrapolate")
    theta_shape = (solvent.multy.shape + r_grid.shape[1:])
    theta = np.zeros(theta_shape)
    for site in solute.sites:
        site_position = np.expand_dims(site.coordinates, axis=(1, 2, 3))
        distances = np.linalg.norm(r_grid - site_position, axis=0)
        theta_site_interpolated = f(distances)
        theta = theta + site.charge * theta_site_interpolated
    return theta 
