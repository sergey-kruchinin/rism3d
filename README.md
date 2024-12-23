# Installation
Run in the package root directory:
`pip install .`

# Usage
```python
import numpy as np
import rism3d

solute = rism3d.Solute.read_from_ambertools("solute.crd", "solute.top")
solvent = rism3d.Solvent.read_from_ambertools("solvent.xvv")
buffer = 15
box = rism3d.Box.around_solute(solute, 0.25 * np.ones(3), buffer)
rism3d_parameters = {"temperature": 298.15,
					 "closure": "kh",
					 "smear": 1,
					}
rism3d_instance = rism3d.Rism3D(solute, solvent, box, rism3d_parameters)
mdiis_parameters = {"nsteps": 10000,
					"accuracy": 1e-4,
					"nvectors": 10,
					"mix": 0.3,
					"max_residue": 100,
				   }
rism3d_solver = rism3d.MDIISSolver(rism3d_instance, mdiis_parameters)
rism3d_solver.solve()

rism3d_instance.get_h()
rism3d_instance.get_c()
