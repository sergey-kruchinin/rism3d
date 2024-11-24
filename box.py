import numpy as np
from scipy import fft


class Box:
    def __init__(self, origin, delta, buffer):
        """Constructor.
        Create rectangular box around origin point.

        Parameters:
            origin : 1d ndarray. Cartesian coordinates of the box 
                     center.
            delta : 1d ndarray. Grid spacings for every of three 
                     dimensions.
            buffer : scalar or 1d ndarray. Space between origin and box 
                     faces for every dimension. When only one value is 
                     supplied it used as common value and resulting Box
                     will be cubic.
        Raises:
            ValueError : When at least one value of buffer is 
                         negative.
        """
        if np.any(buffer < 0):
            raise ValueError("Box buffer value(s) " 
                             + str(buffer) 
                             + " cannot be negative")
        box_dimensions = buffer * 2
        npoints = np.ceil(box_dimensions / delta).astype(int) + 1
        half_widths = (npoints - 1) * delta / 2
        min_bounds = origin - half_widths
        max_bounds = origin + half_widths
        self._r_grid = [np.linspace(m, M, p) for m, M, p 
                        in zip(min_bounds, max_bounds, npoints)]
        self._shape = tuple([len(i) for i in self.r_grid])
        self._k_grid = self._get_k_grid(npoints, delta)
        self._inv_shape = tuple([len(i) for i in self.k_grid])
        self._delta = np.eye(3) * delta
        self._cell_volume = np.prod(delta)
        self._r_grid_prefactor = self._get_r_grid_prefactor()

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
    def x_grid(self):
        return self._r_grid[0]

    @property
    def y_grid(self):
        return self._r_grid[1]

    @property
    def z_grid(self):
        return self._r_grid[2]

    @property
    def shape(self):
        return self._shape

    @property
    def k_grid(self):
        return self._k_grid

    @property
    def x_inv_grid(self):
        return self._k_grid[0]

    @property
    def y_inv_grid(self):
        return self._k_grid[1]

    @property
    def z_inv_grid(self):
        return self._k_grid[2]

    @property
    def shape(self):
        return self._shape

    @property
    def inv_shape(self):
        return self._inv_shape

    @property
    def delta(self):
        return self._delta

    @property
    def cell_volume(self):
        return self._cell_volume

    @property
    def r_grid_prefactor(self):
        return self._r_grid_prefactor

    def select(self, box):
        """Select grid points in real space laying inside the bounds
        of another Box.

        Parameters:
            box : Box instance. Box with bounds for selection.
        Returns:
            3d ndarray : Bool indexing array for r_grid of current 
                         Box defining points laying inside supplying 
                         Box bounds.
        """
        x_selection = np.logical_and(self.x_grid > box.x_grid[0], 
                                     self.x_grid < box.x_grid[-1])
        x_selection = np.expand_dims(x_selection, axis=(1, 2))
        y_selection = np.logical_and(self.y_grid > box.y_grid[0], 
                                     self.y_grid < box.y_grid[-1])
        y_selection = np.expand_dims(y_selection, axis=(1,))
        z_selection = np.logical_and(self.z_grid > box.z_grid[0], 
                                     self.z_grid < box.z_grid[-1])
        selection = np.logical_and(np.logical_and(x_selection, y_selection), 
                                   z_selection)
        return selection

    def _get_k_grid(self, npoints, deltas):
        points_and_steps = [(p, d) for p, d in zip(npoints, deltas)]
        k_grid = [fft.fftfreq(*i) for i in points_and_steps] 
        k_grid[-1] = fft.rfftfreq(*points_and_steps[-1])
        return k_grid

    def _get_r_grid_prefactor(self):
        r0_k = (np.expand_dims(self.x_grid[0] * self.x_inv_grid, axis=(1, 2)) 
                + np.expand_dims(self.y_grid[0] * self.y_inv_grid, axis=(1,)) 
                + self.z_grid[0] * self.z_inv_grid)
        prefactor = np.exp(-2j * np.pi * r0_k)
        return prefactor

    def get_distances(self, origin=np.array([0, 0, 0])):
        d = _calculate_distances(self.r_grid, origin)
        return d

    def get_inv_distances(self, origin=np.array([0, 0, 0])):
        d = _calculate_distances(self.k_grid, origin)
        return d


def _calculate_distances(grid, origin):
    shifted_x = np.expand_dims(grid[0] - origin[0], axis=(1, 2))
    shifted_y = np.expand_dims(grid[1] - origin[1], axis=(1,))
    shifted_z = grid[2] - origin[2]
    d = np.sqrt(shifted_x**2 + shifted_y**2 + shifted_z**2)
    return d
