import numpy as np
from scipy import fft


class Box:
    def __init__(self, origin, deltas, buffer):
        """Constructor.
        Create rectangular box around origin point.

        Parameters:
            origin : 1d ndarray. Cartesian coordinates of the box 
                     center.
            deltas : 1d ndarray. Grid spacings for every of three 
                     dimensions.
            buffer : scalar or 1d ndarray. Space between origin and box 
                     faces for every dimension. When only one value is 
                     supplied it used as common value and resulting Box
                     will be cubic.
        Raises:
            ValueError : When at least one value of buffer is 
                         negative.
        """
        if buffer.any() < 0:
            raise ValueError("Box buffer value(s) " 
                             + str(buffer) 
                             + " cannot be negative")
        box_dimensions = buffer * 2
        npoints = np.ceil(box_dimensions / deltas).astype(int) + 1
        half_widths = (npoints - 1) * deltas / 2
        min_bounds = origin - half_widths
        max_bounds = origin + half_widths
        box = [(m, M, p) for m, M, p in zip(min_bounds, max_bounds, npoints)]
        self._r_grid = self._get_r_grid(box)
        self._k_grid = self._get_k_grid(npoints, deltas)
        self._delta = np.eye(3) * deltas
        self._cell_volume = np.prod(deltas)

    @classmethod
    def around_solute(cls, solute, deltas, buffer):
        """Constructor.
        Create rectangular box around solute.

        Parameters:
            origin : Solute instance. 
            deltas : 1d ndarray. Grid spacings for every of three 
                     dimensions.
            buffer : scalar of 1d ndarray. Space between soluteand box 
                     faces for every dimension. When only one value is 
                     supplied it used as common value.
        """
        origin = solute.get_center()
        half_box_widths = np.max(solute.coordinates, axis=0) + buffer - origin
        return cls(origin, deltas, half_box_widths)

    @classmethod
    def create_from_dx(cls, dx):
        """Constructor.
        Create box from the openDX instance.

        Parameters:
            dx : openDX instance.
        """
        deltas = np.diagonal(dx.delta)
        min_border = dx.origin
        npoints = dx.size
        max_border = (npoints - 1) * deltas + min_border
        origin = (max_border + min_border) / 2
        buffer = max_border - origin
        return cls(origin, deltas, buffer)

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
