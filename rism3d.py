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
        self._solvent = solvent
        self._box = copy.deepcopy(box)
        self._r_grid = self._get_r_grid()
        self._k_grid = self._get_k_grid()
        self._chi = self._get_susceptibility()
        self._solute = solute
        self._v_s = self._get_short_potential()
        self._v_l = self._get_long_potential()
        self._theta = self._get_renormalized_potential()
        self._gamma = np.zeros_like(self._v_s)
        
    @property
    def parameters(self):
        return self._parameters

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
        c_s_ft = self._get_fourier_transform(c_s)
        gamma_ft = np.sum(self._chi 
                          * np.expand_dims(c_s_ft, axis=1),
                          axis=0) - c_s_ft
        self._gamma = self._get_inverse_fourier_transform(gamma_ft)

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
        
    def _get_susceptibility(self):
        k_1d = self._solvent.k_grid
        npoints = len(self._solvent.k_grid)
        chi_1d = self._solvent.susceptibility
        k_3d = np.linalg.norm(self._k_grid, axis=0)
        f = interpolate.interp1d(k_1d, 
                                 chi_1d, 
                                 kind="cubic", 
                                 fill_value="extrapolate")
        chi = f(k_3d)
        return chi 

    def _get_short_potential(self):
        v = (self._get_lj_potential() 
             + self._get_short_electrostatic_potential())
        return v

    def _get_lj_potential(self):
        v = 0
        for site in self._solute.sites:
            site_position = np.expand_dims(site.coordinates, axis=(1, 2, 3))
            d = np.linalg.norm(self._r_grid - site_position, axis=0)
            d[d < 1e-6] = 1e-6
            d = 1.0 / d
            r_min = site.rmin + self._solvent.rmin
            frac = np.tensordot(r_min, d, axes=0)**6
            eps = np.expand_dims(np.sqrt(site.epsilon 
                                         * self._solvent.epsilon),
                                 axis=(1, 2, 3))
            v += self._beta * eps * (frac**2 - 2 * frac)
        return v

    def _get_short_electrostatic_potential(self):
        v = 0
        dieps = self._parameters["dieps"]
        for site in self._solute.sites:
            site_position = np.expand_dims(site.coordinates, axis=(1, 2, 3))
            d = np.linalg.norm(self._r_grid - site_position, axis=0)
            d[d < 1e-6] = 1e-6
            v += site.charge * special.erfc(d * self._parameters["smear"]) / d
        v = np.tensordot(self._solvent.charge, 
                         v, 
                         axes=0) * self._beta / dieps
        return v

    def _get_long_potential(self):
        """Calculate v_long * beta."""
        v_l = 0
        coef = self._beta / self._parameters["dieps"]
        dieps = self._parameters["dieps"]
        for site in self._solute.sites:
            site_position = np.expand_dims(site.coordinates, axis=(1, 2, 3))
            d = np.linalg.norm(self._r_grid - site_position, axis=0)
            d[d < 1e-6] = 1e-6
            v_l += site.charge * special.erf(d * self._parameters["smear"]) / d
        v_l = np.tensordot(self._solvent.charge, 
                           v_l, 
                           axes=0) * self._beta / dieps
        return v_l
        
    def _get_fourier_transform(self, data):
        dV = np.prod(self._get_r_delta())
        shift = np.expand_dims(self._r_grid[:, 0, 0, 0], axis=(1, 2, 3))
        K = np.prod(np.exp(-2j * np.pi * self._k_grid * shift), axis=0)
        inv = fft.rfftn(data, axes=(-3, -2, -1), workers=-1) * dV * K
        return inv

    def _get_inverse_fourier_transform(self, data):
        dV = np.prod(self._get_r_delta())
        shift = np.expand_dims(self._r_grid[:, 0, 0, 0], axis=(1, 2, 3))
        K = np.prod(np.exp(2j * np.pi * self._k_grid * shift), axis=0)
        shape = self._r_grid[0].shape
        inv = fft.irfftn(data * K, s=shape, axes=(-3, -2, -1), workers=-1) / dV
        return inv

    def _get_r_grid(self):
        grids = [np.linspace(*i) for i in self._box]
        r_grid = np.stack(np.meshgrid(*grids, indexing="ij"))
        return r_grid

    def _get_r_delta(self):
        delta = np.array([self._r_grid[0, 1, 0, 0] - self._r_grid[0, 0, 0, 0], 
                          self._r_grid[1, 0, 1, 0] - self._r_grid[1, 0, 0, 0],
                          self._r_grid[2, 0, 0, 1] - self._r_grid[2, 0, 0, 0]])
        return delta

    def _get_k_grid(self):
        points_and_steps = [(i[2], (i[1] - i[0]) / (i[2] - 1)) for i in self._box]
        grids = [fft.fftfreq(*i) for i in points_and_steps] 
        grids[-1] = fft.rfftfreq(*points_and_steps[-1])
        k_grid = np.stack(np.meshgrid(*grids, indexing="ij"))
        return k_grid

    def _get_renormalized_potential(self):
        solvent_charges = self._solvent.charge
        chi = self._solvent.susceptibility
        theta_site_ft = np.einsum("i,ijk->jk", solvent_charges, chi)
        k_1d = self._solvent.k_grid
        phi_ft = (self._beta 
                  * potentials.make_inverse_long_coulomb_potential(k_1d, 1, 1))
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
        theta_shape = (self._solvent.multy.shape 
                       + self._r_grid.shape[1:])
        theta = np.zeros(theta_shape)
        for site in self._solute.sites:
            site_position = np.expand_dims(site.coordinates, axis=(1, 2, 3))
            distances = np.linalg.norm(self._r_grid - site_position, axis=0)
            theta_site_interpolated = f(distances)
            theta = theta + site.charge * theta_site_interpolated
        return theta 
