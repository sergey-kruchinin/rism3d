import math
import numpy as np
import unittest
from mdiis import Solver
from rism3d import Rism3D
from pathlib import Path
from solute import Solute
from solvent import Solvent
from box import Box
from sfe import sfe
import exceptions
from suppress_output import SuppressOutput


class TestMdiisSolver(unittest.TestCase):
    def setUp(self):
        work_dir = Path("integration/fixtures/")
        self.solute = Solute.read_from_ambertools(work_dir / "acetic_acid.crd", 
                                                  work_dir / "acetic_acid.top")
        self.water = Solvent.read_from_ambertools(work_dir / "water.xvv")
        self.box = Box.around_solute(self.solute, 
                                     np.array([0.25, 0.25, 0.25]), 
                                     10)
        self.rism_parameters = {"temperature": 298.15,
                                "smear":       1,
                                "closure":     "hnc",
                               }
        self.solver_parameters = {"nsteps":      1000,
                                  "accuracy":    1e-4,
                                  "nvectors":    10,
                                  "mix":         0.3,
                                  "max_residue": 10}
        
    def test_hnc(self):
        self.rism_parameters["closure"] = "hnc"
        r3d_acetic_acid = Rism3D(self.solute, 
                                 self.water, 
                                 self.box, 
                                 self.rism_parameters)
        acetic_acid_aq_solver = Solver(r3d_acetic_acid, 
                                       self.solver_parameters)
        with SuppressOutput():
            acetic_acid_aq_solver.solve()
        self.assertTrue(math.isclose(sfe(r3d_acetic_acid), 
                                     1.0500847994321265, 
                                     rel_tol=0.05))

    def test_kh(self):
        self.rism_parameters["closure"] = "kh"
        r3d_acetic_acid = Rism3D(self.solute, 
                                 self.water, 
                                 self.box, 
                                 self.rism_parameters)
        acetic_acid_aq_solver = Solver(r3d_acetic_acid, 
                                       self.solver_parameters)
        with SuppressOutput():
            acetic_acid_aq_solver.solve()
        self.assertTrue(math.isclose(sfe(r3d_acetic_acid), 
                                     2.0179017990219053, 
                                     rel_tol=0.05))

    def test_max_steps(self):
        r3d_acetic_acid = Rism3D(self.solute, 
                                 self.water, 
                                 self.box, 
                                 self.rism_parameters)
        self.solver_parameters["nsteps"] = 10
        acetic_acid_aq_solver = Solver(r3d_acetic_acid, 
                                       self.solver_parameters)
        with SuppressOutput():
            with self.assertRaises(exceptions.Rism3DMaxStepError):
                acetic_acid_aq_solver.solve()


if __name__ == "__main__":
    unittest.main()