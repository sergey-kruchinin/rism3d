import numpy as np
import itertools
from numba import jit


class IntramolecularCorrelationMatrix():
    """Create matrix of intramolecular correlations."""

    def __init__(self, solvent):
        sites_in_species = []
        end = 0
        start_index = 0
#        for i in solvent["NVSP"]:
        for i in [len(solvent["multy"]), ]:
            begin = end 
            end = end + i
            number_of_sites = np.sum(solvent["multy"][begin:end])
            stop_index = start_index + number_of_sites
            sites_indexes = list(range(start_index, stop_index))
            start_index = stop_index
            sites_in_species.append(sites_indexes)
        coordinates = solvent["xyz"]
        nsites = coordinates.shape[0]
        self.distances = np.zeros((nsites, nsites))
        for sites in sites_in_species:
            for pair in itertools.permutations(sites, 2):
                self.distances[pair] = np.linalg.norm(coordinates[pair[0]]
                                                      - coordinates[pair[1]])
        np.fill_diagonal(self.distances, 0)

    def at(self, k):
        """Evaluate matrix at point 'k'"""
        close_to_zero = 1e-6
        if k < close_to_zero:
            w = np.ones(self.distances.shape)
        else:
            w = IntramolecularCorrelationMatrix._eval(self.distances.copy(), k)
        return w

    @jit(nopython=True)
    def _eval(distances, k):
        for i in range(distances.shape[0]):
            for j in range(distances.shape[1]):
                if distances[i, j] > 0.01:
                    distances[i, j] *= k
                    distances[i, j] = (np.sin(distances[i, j])
                                       / distances[i, j])
        np.fill_diagonal(distances, 1)
        return distances

