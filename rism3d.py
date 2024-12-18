import itertools
import copy
import numpy as np
from scipy import fft, interpolate
from . import constants, mdiis, exceptions, potentials, fourier


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
        self._theta = _get_long_tcf(self._solute, 
                                    self._solvent, 
                                    self._box, 
                                    self._parameters["smear"], 
                                    self._beta)
        self._gamma = np.zeros_like(self._v_s)
        self._c_s = np.zeros_like(self._v_s)

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        self._gamma = value

    @property
    def closure(self):
        return self._use_closure

    @property
    def oz(self):
        return self._use_oz

    @property
    def parameters(self):
        return self._parameters

    @property
    def box(self):
        return self._box

    @property
    def beta(self):
        return self._beta

    def get_h(self):
        h = self._c_s + self._gamma
        return h

    def get_c(self):
        c = self._c_s - self.beta * self._v_l
        return c

    def _use_oz(self):
        c_s_ft = _get_fourier_transform(self._c_s, self._box)
        gamma_ft = np.sum(self._chi 
                          * np.expand_dims(c_s_ft, axis=1),
                          axis=0) - c_s_ft
        self._gamma = _get_inverse_fourier_transform(gamma_ft, self._box) 
        self._gamma -= self._theta

    def _use_hnc(self):
        self._c_s = (np.exp(-self.beta * self._v_s + self._gamma) 
                     - 1 - self._gamma)

    def _use_kh(self):
        self._c_s = np.zeros_like(self._gamma)
        e = -self.beta * self._v_s + self._gamma
        self._c_s[e > 0] = -self.beta * self._v_s[e > 0]
        self._c_s[e <= 0] = np.exp(e[e <= 0]) - 1 - self._gamma[e <= 0] 

    def _use_pse3(self):
        self._c_s = np.zeros_like(self._gamma)
        e = -self.beta * self._v_s + self._gamma
        self._c_s[e > 0] = (-self._v_s[e > 0] 
                            + (1.0 / 2.0) * e[e > 0]**2 
                            + (1.0 / 6.0) * e[e > 0]**3)
        self._c_s[e <= 0] = np.exp(e[e <= 0]) - 1 - self._gamma[e <= 0] 


def _get_susceptibility(solvent, box):
    k_distances = box.get_inv_distances()
    f = interpolate.interp1d(solvent.k_grid, 
                             solvent.susceptibility, 
                             kind="cubic", 
                             fill_value="extrapolate")
    chi = f(k_distances)
    return chi 


def _get_fourier_transform(data, box):
    dV = box.cell_volume
    inv = (fft.rfftn(data, axes=(-3, -2, -1), workers=-1) 
           * dV 
           * box.r_grid_prefactor)
    return inv


def _get_inverse_fourier_transform(data, box):
    dV = box.cell_volume
    inv = fft.irfftn(data * box._r_grid_prefactor.conjugate(), 
                     s=box.shape, 
                     axes=(-3, -2, -1), 
                     workers=-1) / dV
    return inv


def _get_long_tcf(solute, solvent, box, smear, beta):
    theta_site_ft = np.einsum("i,ijk->jk", solvent.charge, 
                              solvent.susceptibility)
    k_1d = solvent.k_grid
    phi_ft = potentials.get_inverse_long_coulomb(k_1d, 1, 1, smear) * beta 
    theta_site_ft = theta_site_ft * phi_ft
    theta_site_ft_at_0 = interpolate.interp1d(k_1d[1:5], 
                                              theta_site_ft[:, 1:5], 
                                              kind="cubic",
                                              fill_value="extrapolate"
                                             )
    theta_site_ft[:, 0] = theta_site_ft_at_0(0)
    f = interpolate.interp1d(k_1d, 
                             theta_site_ft, 
                             kind="cubic", 
                             fill_value="extrapolate")
    theta_site_ft_interpolated = f(box.get_inv_distances())
    theta_ft_shape = (solvent.multy.shape + box.inv_shape)
    theta_ft = np.zeros(theta_ft_shape)
    for s in solute.sites:
        ri_k = (np.expand_dims(s.coordinates[0] * box.x_inv_grid, axis=(1, 2))
                + np.expand_dims(s.coordinates[1] * box.y_inv_grid, axis=(1,))
                + s.coordinates[2] * box.z_inv_grid)
        site_prefactor = np.exp(-2j * np.pi * ri_k)
        theta_site_ft = site_prefactor * s.charge * theta_site_ft_interpolated
        theta_ft = theta_ft + theta_site_ft
    theta = _get_inverse_fourier_transform(theta_ft, box)
    return theta 
