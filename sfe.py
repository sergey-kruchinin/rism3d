import numpy as np
import warnings


def sfe(rism3d, selection=True):
    sfe_type = {"hnc": _sfe_hnc, "kh": _sfe_kh}
    sfe = sfe_type[rism3d._options["closure"]](rism3d, selection)
    return sfe 


def pressure_correction(rism3d, selection=True):
    pc_type = {"hnc": _pressure_correction_hnc, 
               "kh": _pressure_correction_kh}
    pc = pc_type[rism3d._options["closure"]](rism3d, selection)
    return pc 


def sfe_pc(rism3d, selection=True):
    sfe_pc = sfe(rism3d, selection) + pressure_correction(rism3d, selection)
    return sfe_pc
    

def _sfe_hnc(rism3d, selection=True):
    h = rism3d.h() * selection
    c = rism3d.c() * selection
    rho = rism3d._solvent["density"]
    dV = np.prod(rism3d._calculate_r_delta())
    sfe = np.sum(np.tensordot(rho, 
                              0.5 * h**2 - c - 0.5 * h * c, 
                              axes=1)) * dV / rism3d._beta
    return sfe 


def _sfe_kh(rism3d, selection=True):
    h = rism3d.h() * selection
    c = rism3d.c() * selection
    rho = rism3d._solvent["density"]
    dV = np.prod(rism3d._calculate_r_delta())
    sfe = np.sum(np.tensordot(rho, 
                              (0.5 * h**2 * np.heaviside(-h, 0) 
                               - c 
                               - 0.5 * h * c), 
                              axes=1)) * dV / rism3d._beta
    return sfe


def _pressure_correction_hnc(rism3d, selection=True):
    h = rism3d.h() * selection
    c = rism3d.c() * selection
    rho = rism3d._solvent["density"]
    dV = np.prod(rism3d._calculate_r_delta())
    pc = np.sum(np.tensordot(rho, 
                             0.5 * h + 0.5 * c, 
                             axes=1)) * dV / rism3d._beta
    return pc


def _pressure_correction_kh(rism3d, selection=True):
    pc = _pressure_correction_hnc(rism3d, selection)
    return pc
   

def _calculate_cutoff(rism3d, selection=True):
    """Calculate minimal distances (cutoffs) between the solute atoms 
    and integration area borders.

    Parameters:
        rism3d : Rism3D instance.
        selection : three-dimensional boolean array defining
                    the integration area.
    Returns:
        Array of cutoffs for each atom of the solute.

    If any of the solute atoms is out of the integration area 
    the warning will be thrown.
    """
    xyz_min = np.array([np.min(i[selection]) for i in rism3d._r_grid])
    xyz_max = np.array([np.max(i[selection]) for i in rism3d._r_grid])
    xyz = rism3d._solute["xyz"]
    xyz_cutoffs = np.minimum(xyz - xyz_min, xyz_max - xyz)
    r_cutoffs = np.min(xyz_cutoffs, axis=1)
    if np.any(r_cutoffs < 0):
        warnings.warn("SFE: the solute is out of the integration area")
    return r_cutoffs


def _calculate_dcf_correction(rism3d, selection=True):
    correction = 0
    cutoffs = _calculate_cutoff(rism3d, selection)
    rho = rism3d._solvent["density"]
    dV = np.prod(rism3d._calculate_r_delta())
    selected_grid_points = np.array([i[selection].reshape(-1) 
                                     for i in rism3d._r_grid])
    for xyz, e, r, c in zip(rism3d._solute["xyz"],
                            rism3d._solute["epsilon"],
                            rism3d._solute["rmin"],
                            cutoffs):
        epsilon = np.sqrt(e * rism3d._solvent["epsilon"])
        rmin = r + rism3d._solvent["rmin"]
        distances = np.linalg.norm(selected_grid_points - xyz[:, np.newaxis],
                                   axis=1)
        distances = distances[distances >= c][:, np.newaxis]
#        u = epsilon * ((rmin / distances)**12 - 2 * (rmin / distances)**6)
        u = 0
        tmp = rho * (-np.sum(u) * dV 
                     + np.pi * epsilon * (4 / 9) * (rmin**12 / c**9)
                     - np.pi * epsilon * (8 / 3) * (rmin**6 / c**3))
        correction += np.sum(tmp)
#    correction /= rism3d._beta
    return correction
