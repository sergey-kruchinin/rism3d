import numpy as np
import amberParm
import constants


class SolventSite:
    def __init__(self, charge, rmin, epsilon, density, multy, coordinates):
        self._charge = charge.copy()
        self._rmin = rmin.copy()
        self._epsilon = epsilon.copy()
        self._density = density.copy()
        self._multy = multy.copy()
        self._coordinates = coordinates.copy()
        
    @property
    def charge(self):
        return self._charge

    @property
    def rmin(self):
        return self._rmin

    @property
    def epsilon(self):
        return self._epsilon

    @property
    def density(self):
        return self._density

    @property
    def multy(self):
        return self._multy

    @property
    def coordinates(self):
        return self._coordinates


class SolventIterator:
    def __init__(self, solvent):
        self._solvent = solvent 

    def __iter__(self):
        self._site_index = 0
        self._nonmulty_site_index = 0
        return self

    def __next__(self):
        if self._site_index >= self._solvent.number_of_sites:
            raise StopIteration
        charge = self._solvent.charge[self._site_index]
        rmin = self._solvent.rmin[self._site_index]
        epsilon = self._solvent.epsilon[self._site_index]
        density = self._solvent.density[self._site_index]
        multy = self._solvent.multy[self._site_index]
        first_index = self._nonmulty_site_index
        last_index = first_index + multy
        coordinates = self._solvent.coordinates[first_index:last_index]
        site = SolventSite(charge, rmin, epsilon, density, multy, coordinates)
        self._site_index += 1
        self._nonmulty_site_index += multy
        return site


class Solvent(SolventSite):
    def __init__(self, charge, rmin, epsilon, density, multy, coordinates, 
                 susceptibility, k_grid):
        super().__init__(charge, rmin, epsilon, density, multy, coordinates)
        self._susceptibility = susceptibility
        self._k_grid = k_grid
        self._number_of_sites = len(self._multy)

    @property
    def susceptibility(self):
        return self._susceptibility

    @property
    def k_grid(self):
        return self._k_grid

    @property
    def number_of_sites(self):
        return self._number_of_sites

    @property
    def sites(self):
        sites_iterator = SolventIterator(self)    
        return sites_iterator

    @classmethod
    def read_from_ambertools(cls, xvv_filename):
        xvv = amberParm.amberParm(xvv_filename)
        temperature = xvv["THERMO"][0]
        beta = 1 / (constants.k_Boltzmann * temperature)
        charge = xvv["QV"] / np.sqrt(beta)
        rmin = xvv["RMIN2V"]
        epsilon = xvv["EPSV"] / beta
        density = xvv["RHOV"]
        multy = xvv["MTV"]
        coordinates = xvv["COORD"].reshape((-1, 3))
        npoints = xvv["POINTERS"][0]
        nsites = xvv["POINTERS"][1]
        susceptibility = xvv["XVV"].reshape((nsites, nsites, npoints), 
                                            order="C")
        susceptibility = np.transpose(susceptibility, axes=(1, 0, 2))
        r_delta = xvv["THERMO"][4]
        k_delta = 1 / (2 * npoints * r_delta)
        k_grid = np.arange(npoints) * k_delta
        return cls(charge, rmin, epsilon, density, multy, coordinates, 
                   susceptibility, k_grid)
