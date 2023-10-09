import numpy as np
from scipy import fft


class Box:
    def __init__(self, solute, deltas, buffer):
        """Create box around solute.
        
        Works correctly for centered solutes only.
        """
        size = (np.max(solute.coordinates, axis=0) 
                - np.min(solute.coordinates, axis=0) 
                + 2 * buffer) 
        npoints = np.ceil(size / deltas).astype(int) + 1
        borders = (npoints - 1) * deltas / 2
        box = [(-b, b, p) for b, p in zip(borders, npoints)]
        self._r_grid = self._get_r_grid(box)
        self._k_grid = self._get_k_grid(npoints, deltas)
        self._delta = np.eye(3) * deltas
        self._cell_volume = np.prod(deltas)

#    @classmethod
#    def from_dx(cls, dx):
#        deltas = np.diagonal(dx.delta)
#        min_border = dx.origin
#        npoints = dx.size
#        max_border = (npoints - 1) * deltas + min_border
#        box = [(l, h, p) for l, h, p in zip(min_border, max_border, points)]
#        return box
#
    @property
    def r_grid(self):
        return self._r_grid

    @property
    def k_grid(self):
        return self._k_grid

    @property
    def delta(self):
        return self._delta

    @property
    def cell_volume(self):
        return self._cell_volume

    def _get_r_grid(self, box):
        grids = [np.linspace(*i) for i in box]
        r_grid = np.stack(np.meshgrid(*grids, indexing="ij"))
        return r_grid

    def _get_k_grid(self, npoints, deltas):
        points_and_steps = [(p, d) for p, d in zip(npoints, deltas)]
        grids = [fft.fftfreq(*i) for i in points_and_steps] 
        grids[-1] = fft.rfftfreq(*points_and_steps[-1])
        k_grid = np.stack(np.meshgrid(*grids, indexing="ij"))
        return k_grid
