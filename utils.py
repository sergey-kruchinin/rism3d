import numpy as np
import itertools
import amberParm
import constants


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
