import numpy as np
import itertools
import amberParm
import constants


class BindingEnergy:

    def read_crd(crd_file):
        crd = []
        with open(crd_file, "r") as f:
            f.readline()
            f.readline()
            for line in f:
                data = np.fromstring(line, sep=" ")
                crd.append(data)
        crd = np.hstack(crd).reshape((-1, 3))
#        crd = np.loadtxt(crd_file, skiprows=2).reshape((-1, 3))
        return crd
        
    def read_top(top_file):
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
                             "potential. Processing of this",
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

    def read_solute(crd_file, top_file):
        solute = BindingEnergy.read_top(top_file)
        solute["xyz"] = BindingEnergy.read_crd(crd_file)
        return solute

    def read_solvent(xvv_file):
        xvv = amberParm.amberParm(xvv_file)
        data = {}
        kB_T = constants.k_Boltzmann * xvv["THERMO"][0]
        data["charge"] = xvv["QV"] * np.sqrt(kB_T)
        data["rmin"] = xvv["RMIN2V"]
        data["epsilon"] = xvv["EPSV"] * kB_T
        data["density"] = xvv["RHOV"]
        data["multy"] = xvv["MTV"]
        data["xyz"] = xvv["COORD"].reshape((-1, 3))
        npoints = xvv["POINTERS"][0]
        nsites = xvv["POINTERS"][1]
        data["chi"] = xvv["XVV"].reshape((nsites, nsites, npoints), order="C")
        r_delta = xvv["THERMO"][4]
        k_delta = np.pi / (npoints + 1) / r_delta
        data["k_grid"] = np.arange(npoints) * k_delta
        return data
        
    def read_mdl(mdl_file, molarity):
        def expand(section, multy):
            new = []
            for i, m in zip(section, multy):
                new.append(list(itertools.repeat(i, m)))
            new = np.hstack(new)
            return new
            
        mdl = amberParm.amberParm(mdl_file)
        multy = mdl["MULTI"]
        xyz = mdl["COORD"].reshape((-1, 3))
        charges = expand(mdl["CHG"], multy)
        rmin = expand(mdl["LJSIGMA"], multy)
        epsilon = expand(mdl["LJEPSILON"], multy)
        density = np.full_like(rmin, molarity * constants.N_Avogadro * 1e-27)
        data = {"xyz": xyz,
                "charge": charges, 
                "rmin": rmin, 
                "epsilon": epsilon,
                "multy": np.ones_like(rmin, dtype="int"),
                "density": density,
               }
        return data

    def get_center(crd):
        center = (np.max(crd, axis=0) + np.min(crd, axis=0)) / 2
        return center

    def shift(crd, shift_vector):
        new_crd = crd - shift_vector
        return new_crd

    def create_box(crd, delta, shell):
        """Create box around solute.

        Works correctly for centered solutes only.
        """
        spacing = np.max(crd, axis=0) - np.min(crd, axis=0) + 2 * shell
        npoints = np.ceil(spacing / delta).astype(int) + 1
        borders = (npoints - 1) * delta / 2
        box = [(-b, b, p) for b, p in zip(borders, npoints)]
        return box
