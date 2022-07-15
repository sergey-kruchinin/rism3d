import numpy as np


def sfe(rism3d):
    sfe_type = {"hnc": _sfe_hnc, "kh": _sfe_kh}
    sfe = sfe_type[rism3d._options["closure"]](rism3d)
    return sfe 

def pressure_correction(rism3d):
    pc_type = {"hnc": _pressure_correction_hnc, 
               "kh": _pressure_correction_kh}
    pc = pc_type[rism3d._options["closure"]](rism3d)
    return pc 

def sfe_pc(rism3d):
    sfe_pc = sfe(rism3d) + pressure_correction(rism3d)
    return sfe_pc
    
def _sfe_hnc(rism3d):
    h = rism3d.h()
    c = rism3d.c()
    rho = rism3d._solvent["density"]
    dV = np.prod(rism3d._calculate_r_delta())
    sfe = np.sum(np.tensordot(rho, 
                              0.5 * h**2 - c - 0.5 * h * c, 
                              axes=1)) * dV / rism3d._beta
    return sfe 

def _sfe_kh(rism3d):
    h = rism3d.h()
    c = rism3d.c()
    rho = rism3d._solvent["density"]
    dV = np.prod(rism3d._calculate_r_delta())
    sfe = np.sum(np.tensordot(rho, 
                              (0.5 * h**2 * np.heaviside(-h, 0) 
                               - c 
                               - 0.5 * h * c), 
                              axes=1)) * dV / rism3d._beta
    return sfe

def _pressure_correction_hnc(rism3d):
    h = rism3d.h()
    c = rism3d.c()
    rho = rism3d._solvent["density"]
    dV = np.prod(rism3d._calculate_r_delta())
    pc = np.sum(np.tensordot(rho, 
                             0.5 * h + 0.5 * c, 
                             axes=1)) * dV / rism3d._beta
    return pc

def _pressure_correction_kh(rism3d):
    pc = _pressure_correction_hnc(rism3d)
    return pc
    
