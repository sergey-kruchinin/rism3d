import copy
import numpy as np
from scipy import fft
from scipy import interpolate
from scipy import special
import itertools
import constants
import mdiis
import exceptions


class Rism3D:
    def __init__(self, solute, solvent, box, options):
        self._options = copy.deepcopy(options)
        self._beta = 1 / constants.k_Boltzmann / self._options["temperature"] 
        closures = {"hnc": self._hnc, "kh": self._kh}
        self._closure = closures[self._options["closure"]]
        solvers = {"picard": self._picard_solver, "mdiis": self._mdiis_solver}
        self._solver = solvers[self._options["solver"]]
        self._solvent = copy.deepcopy(solvent)
        self._box = copy.deepcopy(box)
        self._r_grid = self._make_r_grid()
        self._k_grid = self._make_k_grid()
        self._chi = self._calculate_susceptibility()
        self._solute = copy.deepcopy(solute)
        self._v_s = self._calculate_short_potential()
        self._h_l = self._calculate_long_tcf()
        self._v_l = self._calculate_long_potential()
        self._c_s = np.zeros_like(self._v_s)
        self._gamma = np.zeros_like(self._v_s)
        
    def solve(self):
        self._solver()

    def _picard_solver(self):
        mix = self._options["mix"]
        gamma_old = self._gamma.copy()
        step = 0
        print("{0:<6s}{1:>18s}".format("step", "accuracy"))
        while True:
            self._closure()
            self._oz()
            e = np.max(np.abs(self._gamma - gamma_old))
            self._gamma = mix * self._gamma + (1 - mix) * gamma_old
            step += 1
            gamma_old = self._gamma.copy()
            print("{0:<6d}{1:18.8e}".format(step, e))
            if step >= self._options["nsteps"]:
                raise exceptions.Rism3DMaxStepError("The maximum number of steps has been reached", step, e)
            if np.isnan(e) or np.isinf(e):
                raise exceptions.Rism3DConvergenceError("The solution has been diverged", step)
            if e < self._options["accuracy"]:
                self._closure()
                break
                
    def _mdiis_solver(self):
        m = mdiis.MDIIS(self._options["mdiis_vectors"],
                        self._options["mdiis_mix"],
                        self._options["mdiis_max_residue"]
                        )
        gamma_old = self._gamma.copy()
        step = 0
        print("{0:<6s}{1:>18s}{2:>7s}".format("step", "accuracy", "MDIIS"))
        while True:
            self._closure()
            self._oz()
            residual = self._gamma - gamma_old
            self._gamma = m.optimize(gamma_old, residual)
            e = np.max(np.abs(residual))
            step += 1
            gamma_old = self._gamma.copy()
            print(f"{step:<6d}{e:18.8e}{m.size():>7d}")
            if step >= self._options["nsteps"]:
                raise exceptions.Rism3DMaxStepError("The maximum number of steps has been reached", step, e)
            if np.isnan(e) or np.isinf(e):
                raise exceptions.Rism3DConvergenceError("The solution has been diverged", step)
            if e < self._options["accuracy"]:
                self._closure()
                break

    def h(self):
        h = self._c_s + self._gamma + self._h_l
        return h

    def c(self):
        c = self._c_s - self._v_l
        return c

    def _oz(self):
        c_s_ft = self._fourier(self._c_s)
        gamma_ft = np.sum(self._chi 
                          * np.expand_dims(c_s_ft, axis=0),
                          axis=1) - c_s_ft
        self._gamma = self._inverse_fourier(gamma_ft)

    def _hnc(self):
        self._c_s = (np.exp(-self._v_s 
                            + self._h_l 
                            + self._gamma) 
                     - 1 
                     - self._gamma 
                     - self._h_l)

    def _kh(self):
        e = -self._v_s + self._h_l + self._gamma
        self._c_s[e > 0] = -self._v_s[e > 0]
        self._c_s[e <= 0] = (np.exp(e[e <= 0]) 
                            - 1 
                            - self._gamma[e <= 0] 
                            - self._h_l[e <= 0]) 
        
    def _calculate_susceptibility(self):
        k_1d = (self._solvent["k_grid"]) / 2 / np.pi
        npoints = len(self._solvent["k_grid"])
        chi_1d = self._solvent["chi"]
        k_3d = np.linalg.norm(self._k_grid, axis=0)
        f = interpolate.interp1d(k_1d, 
                                 chi_1d, 
                                 kind="cubic", 
                                 fill_value="extrapolate")
        chi = f(k_3d)
        return chi 

    def _calculate_short_potential(self):
        v = (self._calculate_lj_potential() 
             + self._calculate_short_electrostatic_potential())
        return v

    def _calculate_lj_potential(self):
        v = 0
        for r, e, c, in zip(self._solute["rmin"],
                            self._solute["epsilon"],    
                            self._solute["xyz"]):
            d = np.linalg.norm(self._r_grid 
                               - np.expand_dims(c, axis=(1, 2, 3)),
                               axis=0)
            d[d < 1e-6] = 1e-6
            d = 1.0 / d
            r_min = r + self._solvent["rmin"]
            frac = np.tensordot(r_min, d, axes=0)**6
            eps = np.expand_dims(np.sqrt(e * self._solvent["epsilon"]),
                                 axis=(1, 2, 3))
            v += self._beta * eps * (frac**2 - 2 * frac)
        return v

    def _calculate_short_electrostatic_potential(self):
        v = 0
        dieps = self._options["dieps"]
        for q, c in zip(self._solute["charge"], 
                        self._solute["xyz"]):
            d = np.linalg.norm(self._r_grid 
                               - np.expand_dims(c, axis=(1, 2, 3)),
                               axis=0)
            d[d < 1e-6] = 1e-6
            v += q * special.erfc(d * self._options["smear"]) / d
        v = np.tensordot(self._solvent["charge"], 
                         v, 
                         axes=0) * self._beta / dieps
        return v

    def _calculate_long_potential(self):
        """Calculate v_long * beta."""
        v_l = 0
        coef = self._beta / self._options["dieps"]
        dieps = self._options["dieps"]
        for q, c in zip(self._solute["charge"], self._solute["xyz"]):
            d = np.linalg.norm(self._r_grid 
                               - np.expand_dims(c, axis=(1, 2, 3)),
                               axis=0)
            d[d < 1e-6] = 1e-6
            v_l += q * special.erf(d * self._options["smear"]) / d
        v_l = np.tensordot(self._solvent["charge"], 
                           v_l, 
                           axes=0) * self._beta / dieps
        return v_l
        
    def _fourier(self, data):
        dV = np.prod(self._calculate_r_delta())
        shift = np.expand_dims(self._r_grid[:, 0, 0, 0], axis=(1, 2, 3))
        K = np.prod(np.exp(-2j * np.pi * self._k_grid * shift), axis=0)
        inv = fft.rfftn(data, axes=(-3, -2, -1), workers=-1) * dV * K
        return inv

    def _inverse_fourier(self, data):
        dV = np.prod(self._calculate_r_delta())
        shift = np.expand_dims(self._r_grid[:, 0, 0, 0], axis=(1, 2, 3))
        K = np.prod(np.exp(2j * np.pi * self._k_grid * shift), axis=0)
        shape = self._r_grid[0].shape
        inv = fft.irfftn(data * K, s=shape, axes=(-3, -2, -1), workers=-1) / dV
        return inv

    def _calculate_long_tcf(self):
        dieps = self._options["dieps"]
        sum_solute = 0
        for q, c in zip(self._solute["charge"], self._solute["xyz"]):
            r_dot_k = np.tensordot(c, self._k_grid, axes=1)
            sum_solute += q * np.exp(-2j * np.pi * r_dot_k)
        sum_solvent = np.tensordot(self._chi, 
                                   self._solvent["charge"], 
                                   axes=([1, 0]))
        k = np.linalg.norm(self._k_grid, axis=0)
        k[0, 0, 0] = 1
        l = self._options["smear"]
        h_long = (-self._beta / dieps / np.pi / k**2 
                  * np.exp(-np.pi**2 * k**2 / l**2)
                  * sum_solvent
                  * sum_solute)
        h_extrapolate = interpolate.interp1d(k[0, 0, 1:], h_long[:, 0, 0, 1:],
                                             kind="cubic",
                                             fill_value="extrapolate")
        h_long[:, 0, 0, 0] = h_extrapolate(0)
        h_long = self._inverse_fourier(h_long)
        return h_long

    def _extrapolate_long_tcf(self):
        sum_solute = 0
        dieps = self._options["dieps"]
        sum_solute = np.sum(self._solute["charge"])
        sum_solvent = np.tensordot(self._solvent["charge"], 
                                   self._solvent["chi"][:, :, 1:],
                                   axes=1)
        k = self._solvent["k_grid"][1:]
        h_l = -self._beta / dieps / np.pi / k**2 * sum_solute * sum_solvent
        h_extrapolate = interpolate.interp1d(k, h_l, 
                                             kind="cubic", 
                                             fill_value="extrapolate")
        h_l_0 = h_extrapolate(0)
        return h_l_0

    def _make_intramolecular_correlation_matrix(self):
        sites_in_species = []
        end = 0
        start_index = 0
        for i in [len(self._solvent["multy"]), ]:
            begin = end 
            end = end + i
            number_of_sites = np.sum(self._solvent["multy"][begin:end])
            stop_index = start_index + number_of_sites
            sites_indexes = list(range(start_index, stop_index))
            start_index = stop_index
            sites_in_species.append(sites_indexes)
        coordinates = self._solvent["xyz"]
        nsites = coordinates.shape[0]
        distances = np.zeros((nsites, nsites))
        for sites in sites_in_species:
            for pair in itertools.permutations(sites, 2):
                distances[pair] = np.linalg.norm(coordinates[pair[0]]
                                                 - coordinates[pair[1]])
        np.fill_diagonal(distances, 0)
        k = np.linalg.norm(self._k_grid, axis=0)
        # np.pi is reduced in 2pi multiplier cause numpy sinc routine
        # uses definition of sinc as sin(pi x) / (pi x)
        k_distances = np.tensordot(distances, k, axes=0) * 2
        w = np.sinc(k_distances)
        return w

    def _make_r_grid(self):
        grids = [np.linspace(*i) for i in self._box]
        r_grid = np.stack(np.meshgrid(*grids, indexing="ij"))
        return r_grid

    def _calculate_r_delta(self):
        delta = np.array([self._r_grid[0, 1, 0, 0] - self._r_grid[0, 0, 0, 0], 
                          self._r_grid[1, 0, 1, 0] - self._r_grid[1, 0, 0, 0],
                          self._r_grid[2, 0, 0, 1] - self._r_grid[2, 0, 0, 0]])
        return delta

    def _make_k_grid(self):
        points_and_steps = [(i[2], (i[1] - i[0]) / (i[2] - 1)) for i in self._box]
        grids = [fft.fftfreq(*i) for i in points_and_steps] 
        grids[-1] = fft.rfftfreq(*points_and_steps[-1])
        k_grid = np.stack(np.meshgrid(*grids, indexing="ij"))
        return k_grid
