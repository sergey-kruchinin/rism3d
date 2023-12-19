import copy
import numpy as np
from scipy import fft
from scipy import interpolate
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
        self._box = box
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
        self._chi = _get_susceptibility(self._solvent, self._box)
        self._v_s = (potentials.get_lj(self._solute, 
                                       self._solvent, 
                                       self._box) 
                     + potentials.get_short_coulomb(self._solute, 
                                                    self._solvent, 
                                                    self._box, 
                                                    self._parameters["smear"])
                     )
        self._v_l = potentials.get_long_coulomb(self._solute, 
                                                self._solvent, 
                                                self._box, 
                                                self._parameters["smear"]
                                               )
        self._theta = _get_renormalized_potential(self._solute, 
                                                  self._solvent, 
                                                  self._box, 
                                                  self._parameters["smear"], 
                                                  self._beta)
#        self._gamma = np.zeros_like(self._v_s)
        self._step = 0
        self._set_solver()
        
    @property
    def parameters(self):
        return self._parameters

    @property
    def box(self):
        return self._box

    @property
    def beta(self):
        return self._beta

#    def solve(self):
#        self._use_solver()

    def solve(self):
        while True:
            self.iterate()
            e = np.max(np.abs(self._gamma - self._gamma_old))
            print("{0:<6d}{1:18.8e}".format(self._step, e))
            if step >= self._parameters["nsteps"]:
            raise exceptions.Rism3DMaxStepError("The maximum number of steps has been reached", step, e)
        if np.isnan(e) or np.isinf(e):
            raise exceptions.Rism3DConvergenceError("The solution has been diverged", step)
        if e < self._parameters["accuracy"]:
            break

    def get_h(self):
        c_s = self._use_closure()
        h = c_s + self._gamma
        return h

    def get_c(self):
        c_s = self._use_closure()
        c = c_s - self.beta * self._v_l
        return c

    def _set_solver(self):
        setters = {"picard": self._set_picard_solver, 
                   "mdiis": self._use_mdiis_solver}
        setters[self._parameters["solver"]]

    def _set_picard_solver(self):
        self._mix = self._parameters["mix"]
        self._gamma = np.zeros_like(self._v_s)
        self._gamma_old = np.zeros_like(self._v_s)
        print("{0:<6s}{1:>18s}".format("step", "accuracy"))
        self.iterate = self._step_picard
        
    def _step_picard(self):
        c_s = self._use_closure()
        self._use_oz(c_s)
        self._gamma -= self._theta 
        e = np.max(np.abs(self._gamma - self._gamma_old))
        self._gamma = (self._mix * self._gamma 
                       + (1 - self._mix) * self._gamma_old)
        self._step += 1
        self._gamma_old = self._gamma.copy()
        print("{0:<6d}{1:18.8e}".format(step, e))
        if step >= self._parameters["nsteps"]:
            raise exceptions.Rism3DMaxStepError("The maximum number of steps has been reached", step, e)
        if np.isnan(e) or np.isinf(e):
            raise exceptions.Rism3DConvergenceError("The solution has been diverged", step)
        if e < self._parameters["accuracy"]:
            break
        
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
        c_s_ft = _get_fourier_transform(c_s, self._box)
        gamma_ft = np.sum(self._chi 
                          * np.expand_dims(c_s_ft, axis=1),
                          axis=0) - c_s_ft
        self._gamma = _get_inverse_fourier_transform(gamma_ft, self._box)

    def _use_hnc(self):
        c_s = np.exp(-self.beta * self._v_s + self._gamma) - 1 - self._gamma
        return c_s

    def _use_kh(self):
        c_s = np.zeros_like(self._gamma)
        e = -self.beta * self._v_s + self._gamma
        c_s[e > 0] = -self.beta * self._v_s[e > 0]
        c_s[e <= 0] = np.exp(e[e <= 0]) - 1 - self._gamma[e <= 0] 
        return c_s

    def _use_pse3(self):
        c_s = np.zeros_like(self._gamma)
        e = -self.beta * self._v_s + self._gamma
        c_s[e > 0] = (-self._v_s[e > 0] 
                      + (1.0 / 2.0) * e[e > 0]**2 
                      + (1.0 / 6.0) * e[e > 0]**3)
        c_s[e <= 0] = np.exp(e[e <= 0]) - 1 - self._gamma[e <= 0] 
        return c_s


def _get_susceptibility(solvent, box):
    k_distances = np.linalg.norm(box.k_grid, axis=0)
    f = interpolate.interp1d(solvent.k_grid, 
                             solvent.susceptibility, 
                             kind="cubic", 
                             fill_value="extrapolate")
    chi = f(k_distances)
    return chi 


def _get_fourier_transform(data, box):
    dV = box.cell_volume
    shift = np.expand_dims(box.r_grid[:, 0, 0, 0], axis=(1, 2, 3))
    K = np.prod(np.exp(-2j * np.pi * box.k_grid * shift), axis=0)
    inv = fft.rfftn(data, axes=(-3, -2, -1), workers=-1) * dV * K
    return inv


def _get_inverse_fourier_transform(data, box):
    dV = box.cell_volume
    shift = np.expand_dims(box.r_grid[:, 0, 0, 0], axis=(1, 2, 3))
    K = np.prod(np.exp(2j * np.pi * box.k_grid * shift), axis=0)
    shape = box.r_grid[0].shape
    inv = fft.irfftn(data * K, s=shape, axes=(-3, -2, -1), workers=-1) / dV
    return inv


def _get_renormalized_potential(solute, solvent, box, smear, beta):
    theta_site_ft = np.einsum("i,ijk->jk", solvent.charge, 
                              solvent.susceptibility)
    k_1d = solvent.k_grid
    phi_ft = potentials.get_inverse_long_coulomb(k_1d, 1, 1, smear) * beta 
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
    theta_shape = (solvent.multy.shape + box.r_grid.shape[1:])
    theta = np.zeros(theta_shape)
    for site in solute.sites:
        site_position = np.expand_dims(site.coordinates, axis=(1, 2, 3))
        distances = np.linalg.norm(box.r_grid - site_position, axis=0)
        theta_site_interpolated = f(distances)
        theta = theta + site.charge * theta_site_interpolated
    return theta 
