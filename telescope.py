import numpy as np
from scipy import fft
from scipy import interpolate
from scipy import special
import itertools
import constants


class Telescope:
    def __init__(self, solute, solvent, pocket, periphery, rest, options):
        self._options = options
        self._beta = 1 / constants.k_Boltzmann / self._options["temperature"] 
        self._solvent = solvent
        self._r_grid = self._make_r_grid(pocket)
        self._k_grid = self._make_k_grid(pocket)
        self._chi = self._calculate_susceptibility()
        [self._pocket_atoms, 
         self._periphery_atoms, 
         self._rest_atoms] = self._split_solute(solute, pocket, periphery)
        self._v_s = self._calculate_short_potential()
        self._h_l = self._calculate_long_tcf(self._pocket_atoms)
        self._v_l = self._calculate_long_potential(self._pocket_atoms)
        

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

    def _solve_3drism(self):
        gamma_0 = np.zeros_like(self._v_s)
        step = 0
        print("{0:<6s}{1:>18s}".format("step", "accuracy"))
        while True:
            c_s = self._closure(gamma_0, self._h_l)
            c_s_ft = self._fourier(c_s)
            gamma_1_ft = self._oz(c_s_ft)
            gamma_1 = (self._options["mix"] 
                       * self._inverse_fourier(gamma_1_ft)
                       + (1 - self._options["mix"])
                       * gamma_0)
            e = np.mean(np.abs(gamma_1 - gamma_0))
            step += 1
            gamma_0 = gamma_1
            print("{0:<6d}{1:18.8e}".format(step, e))
            if e < self._options["accuracy"] or step >= self._options["nsteps"]:
                c_s = self._closure(gamma_0, self._h_l)
                break
        return gamma_0, c_s

    def _oz(self, c_s_ft):
        gamma_ft = np.sum(self._chi * np.expand_dims(c_s_ft, axis=0), axis=1)
        gamma_ft = gamma_ft - c_s_ft
        return gamma_ft

    def _closure(self, gamma_s, h_l):
        c_s = (np.exp(-self._v_s + h_l + gamma_s) - 1 - gamma_s - h_l)
        return c_s
        
    def _calculate_short_potential(self):
        v = (self._calculate_lj_potential() 
             + self._calculate_short_electrostatic_potential())
        return v

    def _calculate_lj_potential(self):
        v = 0
        for r, e, c, in zip(self._pocket_atoms["rmin"],
                            self._pocket_atoms["epsilon"],    
                            self._pocket_atoms["xyz"]):
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
        for q, c in zip(self._pocket_atoms["charge"], 
                        self._pocket_atoms["xyz"]):
            d = np.linalg.norm(self._r_grid 
                               - np.expand_dims(c, axis=(1, 2, 3)),
                               axis=0)
            d[d < 1e-6] = 1e-6
            v += q * special.erfc(d * self._options["smear"]) / d
        v = np.tensordot(self._solvent["charge"], 
                         v, 
                         axes=0) * self._beta / dieps
        return v

    def _calculate_long_potential(self, atoms):
        """Calculate v_long * beta."""
        v_l = 0
        coef = self._beta / self._options["dieps"]
        dieps = self._options["dieps"]
        if atoms["xyz"].size:
            for q, c in zip(atoms["charge"], atoms["xyz"]):
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

    def _calculate_long_tcf(self, atoms):
        dieps = self._options["dieps"]
        sum_solute = 0
        if atoms["xyz"].size:
            for q, c in zip(atoms["charge"], atoms["xyz"]):
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

    def _extrapolate_long_tcf(self, atoms):
        sum_solute = 0
        dieps = self._options["dieps"]
        if atoms["charge"].size:
            sum_solute = np.sum(atoms["charge"])
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

    def _make_r_grid(self, box):
        grids = [np.linspace(*i) for i in box]
        r_grid = np.stack(np.meshgrid(*grids, indexing="ij"))
        return r_grid

    def _calculate_r_delta(self):
        delta = np.array([self._r_grid[0, 1, 0, 0] - self._r_grid[0, 0, 0, 0], 
                          self._r_grid[1, 0, 1, 0] - self._r_grid[1, 0, 0, 0],
                          self._r_grid[2, 0, 0, 1] - self._r_grid[2, 0, 0, 0]])
        return delta

    def _make_k_grid(self, box):
        points_and_steps = [(i[2], (i[1] - i[0]) / (i[2] - 1)) for i in box]
        grids = [fft.fftfreq(*i) for i in points_and_steps] 
        grids[-1] = fft.rfftfreq(*points_and_steps[-1])
        k_grid = np.stack(np.meshgrid(*grids, indexing="ij"))
        return k_grid

    def _split_solute(self, solute, pocket, periphery):
        keys_list = solute.keys()
        pocket_atoms = {k: [] for k in keys_list}
        periphery_atoms = {k: [] for k in keys_list}
        rest_atoms = {k: [] for k in keys_list}
        pocket_l_borders = np.array([i[0] for i in pocket])
        pocket_h_borders = np.array([i[1] for i in pocket])
        periphery_l_borders = np.array([i[0] for i in periphery])
        periphery_h_borders = np.array([i[1] for i in periphery])
        for i, c in enumerate(solute["xyz"]):
            if (np.all(c >= pocket_l_borders) 
                and np.all(c <= pocket_h_borders)):    
                for k in keys_list:
                    pocket_atoms[k].append(solute[k][i])
                continue
            elif (np.all(c >= periphery_l_borders) 
                  and np.all(c <= periphery_h_borders)):
                for k in keys_list:
                    periphery_atoms[k].append(solute[k][i])
                continue
            else:
                for k in keys_list:
                    rest_atoms[k].append(solute[k][i])
        for k in keys_list:
            pocket_atoms[k] = np.array(pocket_atoms[k])
            periphery_atoms[k] = np.array(periphery_atoms[k])
            rest_atoms[k] = np.array(rest_atoms[k])
        return pocket_atoms, periphery_atoms, rest_atoms
