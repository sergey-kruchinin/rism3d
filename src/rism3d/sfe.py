import numpy as np
import warnings
import copy


def sfe(rism3d, selection=True, dcf_correction=True):
    """Calculate solvation free energy.

    Parameters:
        rism3d : Rism3D instance with converged solution.
        selection : three-dimensional boolean array defining
                    the integration area.
    Returns:
        Solvation free energy value.
    """
    sfe_type = {"hnc": _sfe_hnc, "kh": _sfe_kh}
    sfe = sfe_type[rism3d.parameters["closure"]](rism3d, selection, 
                                                 dcf_correction)
    return sfe 


def pressure_correction(rism3d, selection=True):
    """Calculate pressure correction for solvation free energy.

    Parameters:
        rism3d : Rism3D instance with converged solution.
        selection : three-dimensional boolean array defining
                    the integration area.
    Returns:
        Solvation free energy value.
    """
    pc_type = {"hnc": _pressure_correction_hnc, 
               "kh": _pressure_correction_kh}
    pc = pc_type[rism3d.parameters["closure"]](rism3d, selection)
    return pc 


def sfe_pc(rism3d, selection=True, dcf_correction=True):
    """Calculate solvation free energy with pressure correction.

    Parameters:
        rism3d : Rism3D instance with converged solution.
        selection : three-dimensional boolean array defining
                    the integration area.
    Returns:
        Solvation free energy value.
    """
    sfe_pc = (sfe(rism3d, selection, dcf_correction) 
              + pressure_correction(rism3d, selection))
    return sfe_pc
    

def _sfe_hnc(rism3d, selection=True, dcf_correction=True):
    """Calculate solvation free energy for HNC closure.

    Parameters:
        rism3d : Rism3D instance with converged solution.
        selection : three-dimensional boolean array defining
                    the integration area.
    Returns:
        Solvation free energy value.
    """
    h = rism3d.get_h() * selection
    c = rism3d.get_c() * selection
    rho = rism3d._solvent.density
    dV = rism3d.box.cell_volume
    dcf_corrections = 0
    if dcf_correction:
        dcf_corrections = _calculate_dcf_corrections(rism3d, selection)
    integrals = (np.sum(0.5 * h**2 - c - 0.5 * h * c, axis=(1, 2, 3)) * dV
                 - dcf_corrections)
    sfe = np.sum(integrals * rho) / rism3d.beta
    return sfe 


def _sfe_kh(rism3d, selection=True, dcf_correction=True):
    """Calculate solvation free energy for KH closure.

    Parameters:
        rism3d : Rism3D instance with converged solution.
        selection : three-dimensional boolean array defining
                    the integration area.
    Returns:
        Solvation free energy value.
    """
    h = rism3d.get_h() * selection
    c = rism3d.get_c() * selection
    rho = rism3d._solvent.density
    dV = rism3d.box.cell_volume
    dcf_corrections = 0
    if dcf_correction:
        dcf_corrections = _calculate_dcf_corrections(rism3d, selection)
    integrals = (np.sum(0.5 * h**2 * np.heaviside(-h, 0) - c - 0.5 * h * c, 
                        axis=(1, 2, 3)) * dV
                 - dcf_corrections)
    sfe = np.sum(integrals * rho) / rism3d.beta
    return sfe


def _pressure_correction_hnc(rism3d, selection=True):
    """Calculate pressure correction for solvation free energy 
    for HNC closure. 

    Parameters:
        rism3d : Rism3D instance with converged solution.
        selection : three-dimensional boolean array defining
                    the integration area.
    Returns:
        Solvation free energy value.
    """
    h = rism3d.get_h() * selection
    c = rism3d.get_c() * selection
    rho = rism3d._solvent.density
    dV = rism3d.box.cell_volume
    pc = np.sum(np.tensordot(rho, 
                             0.5 * h + 0.5 * c, 
                             axes=1)) * dV / rism3d.beta
    return pc


def _pressure_correction_kh(rism3d, selection=True):
    """Calculate pressure correction for solvation free energy 
    for KH closure. 

    Parameters:
        rism3d : Rism3D instance with converged solution.
        selection : three-dimensional boolean array defining
                    the integration area.
    Returns:
        Solvation free energy value.
    """
    pc = _pressure_correction_hnc(rism3d, selection)
    return pc
   

def _calculate_cutoff(rism3d, selection=True):
    """Calculate minimal distances (cutoffs) between the solute atoms 
    and integration area borders.

    Parameters:
        rism3d : Rism3D instance with converged solution.
        selection : three-dimensional boolean array defining
                    the integration area.
    Returns:
        Array of cutoffs for each atom of the solute.

    If any of the solute atoms is out of the integration area 
    the warning will be thrown.
    """
    selection = np.logical_and(selection, np.full(rism3d.box.shape, True))
    x_sel = rism3d.box.x_grid[np.any(selection, axis=(1, 2))]
    y_sel = rism3d.box.y_grid[np.any(selection, axis=(0, 2))]
    z_sel = rism3d.box.z_grid[np.any(selection, axis=(0, 1))]
    xyz_min = np.array([np.min(i) for i in [x_sel, y_sel, z_sel]])
    xyz_max = np.array([np.max(i) for i in [x_sel, y_sel, z_sel]])
    xyz = rism3d._solute.coordinates
    xyz_cutoffs = np.minimum(xyz - xyz_min, xyz_max - xyz)
    r_cutoffs = np.min(xyz_cutoffs, axis=1)
    if np.any(r_cutoffs < 0):
        warnings.warn("SFE: the solute is out of the integration area")
    return r_cutoffs


#def _calculate_dcf_corrections(rism3d, selection=True):
#    """Calculate correction to integral of direct correlation function 
#    needed to fix box truncation effect.
#
#    Parameters:
#        rism3d : Rism3D instance with converged solution.
#        selection : three-dimensional boolean array defining
#                    the integration area.
#    Returns:
#        Array of correction values for each atom of solvent.
#    """
#    corrections = np.zeros(rism3d._solvent.number_of_sites)
#    cutoffs = _calculate_cutoff(rism3d, selection)
#    dV = rism3d.box.cell_volume
#    selected_grid_points = copy.deepcopy(rism3d.box.r_grid).reshape((3, -1))
#    for site, cut in zip(rism3d._solute.sites, cutoffs):
#        epsilon = np.sqrt(site.epsilon * rism3d._solvent.epsilon)
#        rmin = site.rmin + rism3d._solvent.rmin
#        site_position = site.coordinates[:, np.newaxis]
#        distances = np.linalg.norm(selected_grid_points 
#                                   - site_position, axis=0)
#        distances = distances[distances >= cut][:, np.newaxis]
#        u = epsilon * ((rmin / distances)**12 - 2 * (rmin / distances)**6)
#        corrections += (np.sum(u, axis=0) * dV
#                        - np.pi * epsilon * (4 / 9) * (rmin**12 / cut**9)
#                        + np.pi * epsilon * (8 / 3) * (rmin**6 / cut**3))
#    corrections *= rism3d.beta
#    return corrections
#
#
def _calculate_dcf_corrections(rism3d, selection=True):
    """Calculate correction to integral of direct correlation function 
    needed to fix box truncation effect.

    Parameters:
        rism3d : Rism3D instance with converged solution.
        selection : three-dimensional boolean array defining
                    the integration area.
    Returns:
        Array of correction values for each atom of solvent.
    """
    corrections = np.zeros(rism3d._solvent.number_of_sites)
    cutoffs = _calculate_cutoff(rism3d, selection)
    dV = rism3d.box.cell_volume
    for site, cut in zip(rism3d._solute.sites, cutoffs):
        epsilon = np.sqrt(site.epsilon * rism3d._solvent.epsilon)
        rmin = site.rmin + rism3d._solvent.rmin
        distances = rism3d.box.get_distances(site.coordinates).reshape(-1)
        distances = distances[distances >= cut][:, np.newaxis]
        u = epsilon * ((rmin / distances)**12 - 2 * (rmin / distances)**6)
        corrections += (np.sum(u, axis=0) * dV
                        - np.pi * epsilon * (4 / 9) * (rmin**12 / cut**9)
                        + np.pi * epsilon * (8 / 3) * (rmin**6 / cut**3))
    corrections *= rism3d.beta
    return corrections
