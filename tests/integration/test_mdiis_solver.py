import math
import numpy as np
import unittest
from pathlib import Path
import rism3d
from suppress_output import SuppressOutput


class TestMdiisSolver(unittest.TestCase):
    def setUp(self):
        work_dir = Path("integration/fixtures/")
        crd_filename = work_dir / "acetic_acid.crd"
        top_filename = work_dir / "acetic_acid.top"
        self.solute = rism3d.Solute.read_from_ambertools(crd_filename, 
                                                         top_filename)
        solvent_filename = work_dir / "water.xvv"
        self.water = rism3d.Solvent.read_from_ambertools(solvent_filename)
        self.box = rism3d.Box.around_solute(self.solute, 
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
        r3d_acetic_acid = rism3d.Rism3D(self.solute, 
                                        self.water, 
                                        self.box, 
                                        self.rism_parameters)
        acetic_acid_aq_solver = rism3d.MDIISSolver(r3d_acetic_acid, 
                                                   self.solver_parameters)
        with SuppressOutput():
            acetic_acid_aq_solver.solve()
        self.assertTrue(math.isclose(rism3d.sfe.sfe(r3d_acetic_acid), 
                                     1.0500847994321265, 
                                     rel_tol=0.05))

    def test_kh(self):
        self.rism_parameters["closure"] = "kh"
        r3d_acetic_acid = rism3d.Rism3D(self.solute, 
                                        self.water, 
                                        self.box, 
                                        self.rism_parameters)
        acetic_acid_aq_solver = rism3d.MDIISSolver(r3d_acetic_acid, 
                                                   self.solver_parameters)
        with SuppressOutput():
            acetic_acid_aq_solver.solve()
        self.assertTrue(math.isclose(rism3d.sfe.sfe(r3d_acetic_acid), 
                                     2.0179017990219053, 
                                     rel_tol=0.05))

    def test_max_steps(self):
        r3d_acetic_acid = rism3d.Rism3D(self.solute, 
                                        self.water, 
                                        self.box, 
                                        self.rism_parameters)
        self.solver_parameters["nsteps"] = 10
        acetic_acid_aq_solver = rism3d.MDIISSolver(r3d_acetic_acid, 
                                                   self.solver_parameters)
        with SuppressOutput():
            with self.assertRaises(rism3d.exceptions.Rism3DMaxStepError):
                acetic_acid_aq_solver.solve()


if __name__ == "__main__":
    unittest.main()
