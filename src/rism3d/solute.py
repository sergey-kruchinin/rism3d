import numpy as np
from . import amberParm


class SoluteSite:
    def __init__(self, charge, rmin, epsilon, coordinates):
        self._charge = charge.copy()
        self._rmin = rmin.copy()
        self._epsilon = epsilon.copy()
        self._coordinates = coordinates.copy()
        
    @property
    def charge(self):
        return self._charge

    @charge.setter
    def charge(self, value):
        self._charge = value

    @property
    def rmin(self):
        return self._rmin

    @rmin.setter
    def rmin(self, value):
        self._rmin = value

    @property
    def epsilon(self):
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value):
        self._epsilon = value

    @property
    def coordinates(self):
        return self._coordinates

    @coordinates.setter
    def coordinates(self, value):
        self._coordinates = value


class SoluteIterator:
    def __init__(self, solute):
        self._solute = solute

    def __iter__(self):
        self._site_index = 0
        return self

    def __next__(self):
        if self._site_index >= self._solute.number_of_sites:
            raise StopIteration
        site = SoluteSite(self._solute.charge[self._site_index],
                          self._solute.rmin[self._site_index],
                          self._solute.epsilon[self._site_index],
                          self._solute.coordinates[self._site_index])
        self._site_index += 1
        return site


class Solute(SoluteSite):
    def __init__(self, charge, rmin, epsilon, coordinates):
        super().__init__(charge, rmin, epsilon, coordinates)
        self._number_of_sites = len(self._charge)

    @property
    def number_of_sites(self):
        return self._number_of_sites

    @property
    def sites(self):
        sites_iterator = SoluteIterator(self)    
        return sites_iterator

    @classmethod
    def read_from_ambertools(cls, crd_filename, top_filename):
        top_data = _read_top(top_filename)
        charge = top_data["charge"]
        rmin = top_data["rmin"]
        epsilon = top_data["epsilon"]
        coordinates = _read_crd(crd_filename)
        return cls(charge, rmin, epsilon, coordinates) 

    def get_center(self):
        center = (np.max(self.coordinates, axis=0) 
                  + np.min(self.coordinates, axis=0)) / 2
        return center

    def shift(self, shift_vector):
        self._coordinates -= shift_vector

    def shift_to_center(self):
        """Shift solute to its center"""
        shift_vector = self.get_center()
        self.shift(shift_vector)


def _read_top(top_file):
    top = amberParm.amberParm(top_file)
    charges = top["CHARGE"]
    A = top["LENNARD_JONES_ACOEF"][
            top["NONBONDED_PARM_INDEX"][
            np.max(top["ATOM_TYPE_INDEX"])
            * (top["ATOM_TYPE_INDEX"] - 1) 
            + top["ATOM_TYPE_INDEX"] - 1] - 1]
    B = top["LENNARD_JONES_BCOEF"][
            top["NONBONDED_PARM_INDEX"][
            np.max(top["ATOM_TYPE_INDEX"])
            * (top["ATOM_TYPE_INDEX"] - 1)
            + top["ATOM_TYPE_INDEX"] - 1] - 1]
    A_is_zero = A < 1e-6
    B_is_zero = B < 1e-6
    if np.bitwise_xor(A_is_zero, B_is_zero).any():
        raise ValueError("Some solute atoms have only repulsion ", 
                         "or attraction terms of Lennard-Jones ", 
                         "potential. Processing of this ",
                         "situation is not implemented yet.")
    epsilon = np.zeros(A.shape)
    rmin2 = np.zeros(A.shape)
    A_and_B_not_zero = np.invert(A_is_zero) & np.invert(B_is_zero)
    epsilon[A_and_B_not_zero] = (B[A_and_B_not_zero]**2 
                                 / 4 
                                 / A[A_and_B_not_zero])
    rmin2[A_and_B_not_zero] = (2 * A[A_and_B_not_zero] 
                               / B[A_and_B_not_zero])**(1.0/6) / 2
    parameters = {"charge": charges, 
                  "rmin": rmin2,
                  "epsilon": epsilon
                 }
    return parameters


def _read_crd(crd_file):
    """
    Read .crd file from AmberTools
    """
    crd = []
    with open(crd_file, "r") as f:
        f.readline()
        f.readline()
        for line in f:
            data = np.fromstring(line, sep=" ")
            crd.append(data)
    crd = np.hstack(crd).reshape((-1, 3))
    return crd
