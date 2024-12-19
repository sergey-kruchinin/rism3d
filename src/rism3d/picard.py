import numpy as np
from . import exceptions


class Solver:
    def __init__(self, rism3d_instance, parameters):
        self._step = 0
        self._rism3d_instance = rism3d_instance
        self._gamma_0 = np.zeros_like(self._rism3d_instance.gamma)
        self._use_closure = self._rism3d_instance.closure
        self._use_oz = self._rism3d_instance.oz
        self._nsteps = parameters["nsteps"]
        self._mix = parameters["mix"]
        self._accuracy = parameters["accuracy"]

    def iterate(self):
        self._step += 1
        self._gamma_0 = self._rism3d_instance.gamma.copy()
        self._use_closure()
        self._use_oz()
        self._rism3d_instance.gamma = (self._mix * self._rism3d_instance.gamma 
                                       + (1 - self._mix) * self._gamma_0)
        if self.step >= self._nsteps:
            error_message = "The maximum number of steps has been reached" 
            raise exceptions.Rism3DMaxStepError(error_message, self.step)

    def solve(self):
        print("{0:<6s}{1:>18s}".format("step", "accuracy"))
        while True:
            self.iterate()
            e = np.max(np.abs(self._rism3d_instance.gamma - self._gamma_0))
            print("{0:<6d}{1:18.8e}".format(self.step, e))
            if np.isnan(e) or np.isinf(e):
                error_message ="The solution has been diverged" 
                raise exceptions.Rism3DConvergenceError(error_message, 
                                                        self.step)
            if e < self._accuracy:
                break

    @property
    def step(self):
        return self._step
