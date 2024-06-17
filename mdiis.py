import warnings
from collections import deque
import numpy as np
from scipy import linalg
import exceptions
warnings.filterwarnings("error",
                        category=linalg.LinAlgWarning,
                        module="mdiis")


class Solver:
    def __init__(self, rism3d_instance, parameters):
        self._step = 0
        self._rism3d_instance = rism3d_instance
        self._gamma_0 = np.zeros_like(self._rism3d_instance.gamma)
        self._use_closure = self._rism3d_instance.closure
        self._use_oz = self._rism3d_instance.oz
        self._nsteps = parameters["nsteps"]
        self._accuracy = parameters["accuracy"]
        self._m = MDIIS(parameters["nvectors"],
                        parameters["mix"],
                        parameters["max_residue"]
                       )

    def iterate(self):
        self._step += 1
        self._gamma_0 = self._rism3d_instance.gamma.copy()
        self._use_closure()
        self._use_oz()
        residual = self._rism3d_instance.gamma - self._gamma_0
        self._rism3d_instance.gamma = self._m.optimize(self._gamma_0, residual)
        if self.step >= self._nsteps:
            error_message = "The maximum number of steps has been reached"
            raise exceptions.Rism3DMaxStepError(error_message, self.step)

    def solve(self):
        print("{0:<6s}{1:>18s}{2:>7s}".format("step", "accuracy", "MDIIS"))
        while True:
            self.iterate()
            e = np.max(np.abs(self._rism3d_instance.gamma - self._gamma_0))
            print(f"{self.step:<6d}{e:18.8e}{self._m.size():>7d}")
            if np.isnan(e) or np.isinf(e):
                error_message = "The solution has been diverged"
                raise exceptions.Rism3DConvergenceError(error_message, step)
            if e < self._accuracy:
                break

    @property
    def step(self):
        return self._step

class MDIIS:
    """Implements the MDIIS algorithm 
    [Kovalenko A. et. al., J. Comput. Chem., 1999, 20, 928]

    Usage:
        m = MDIIS()
        approximation = m.optimize(new_vector, new_residue)

    Parameters:
        N - maximal number of MDIIS vectors
        mix - mixing parameter
        r_max - restart parameter. If 
                R_{new} > R_{min} * r_max MDIIS restarts

    Methods:
        optimize(v, r) - take new vector (v) and residue (r)
                         and return optimized solution
        size() - return current number of MDIIS vectors
    """
    def __init__(self, N=10, mix=0.3, r_max=10):
        self._mix = mix
        self._r_max = r_max
        self._N = N
        self._vectors = deque([], self._N)
        self._residuals = deque([], self._N)
        self._matrix = np.zeros((1, 1))
        self._residual_norms = deque([], self._N)
        self._minimal_norm = np.infty
        self._b = np.full(1, -1.0)

    def optimize(self, v, r):
        """Take new vector (v) and residue (r) 
        and return optimized solution.
        """
        current_norm = self._get_residual_rms_norm(r)
        if current_norm > self._r_max * self._minimal_norm:
            v_min = self._minimal_vector()
            self._flush()
            return v_min
        self._add(v, r)
        try:
            coefficients = linalg.solve(self._matrix, self._b, assume_a="sym")
        except (linalg.LinAlgError, linalg.LinAlgWarning):
            v_min = self._minimal_vector()
            self._flush()
            return v_min
        v_new = (np.tensordot(coefficients[:-1], self._vectors, axes=1) 
                 + self._mix * np.tensordot(coefficients[:-1], self._residuals, axes=1))
        return v_new

    def size(self):
        """Return current number of MDIIS vectors.
        """
        return len(self._vectors)
    
    def _get_residual_rms_norm(self, x):
        """Calculate residual RMS norm.
        """
        rms_norm = np.sum([np.linalg.norm(r)**2 for r in self._residuals])
        rms_norm += np.linalg.norm(x)**2
        new_mdiis_size = self.size() + 1
        if self.size == self._N:
            rms_norm -= np.linalg.norm(self._residuals[-1])
            new_mdiis_size = self.size()
        rms_norm /= (x.size * new_mdiis_size)
        rms_norm = np.sqrt(rms_norm)
        return rms_norm

    def _minimal_vector(self):
        """Return MDIIS vector with minimal norm.
        """
        minimal_pos = np.argmin(self._residual_norms)
        return self._vectors[minimal_pos]

    def _update_matrix(self):
        """Update matrix of coefficients after adding new residue.
        """
        size = self.size()
        tmp = np.full((size+1, size+1), -1.0)
        tmp[1:-1, 1:-1] = self._matrix[:size-1, :size-1]
        tmp[0, :size] = [np.sum(self._residuals[0] * i) 
                         for i in self._residuals]
        tmp[:, 0] = tmp[0, :]
        tmp[-1, -1] = 0
        self._matrix = tmp
        self._b = np.zeros(size+1)
        self._b[-1] = -1

    def _add(self, v, r):
        """Add new vector and residue and update MDIIS state.
        """
        self._vectors.appendleft(v)
        self._residuals.appendleft(r)
        self._residual_norms.appendleft(self._get_residual_rms_norm(r))
        self._minimal_norm = np.min(self._residual_norms)
        self._update_matrix()

    def _flush(self):
        """Reset MDIIS state. Delete all MDIIS vectors and residues.
        """
        size = self.size()
        self._vectors.clear()
        self._residuals.clear()
        self._residual_norms.clear()
        self._matrix = np.zeros((1, 1))
        self._minimal_norm = np.infty
        self._b = np.full(1, -1.0)
