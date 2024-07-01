cimport cython
import numpy as np
cimport numpy as cnp
from src.metrics import distance_metrics
from src.metrics cimport distance_metrics

cnp.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double[:] find_radius(double[:,:] X, cnp.int64_t[:] medoids_idx, str metric):
    cdef int M = medoids_idx.shape[0]
    cdef double d
    cdef double[:] radius = np.zeros(M, dtype=np.float64)

    for i in range(M):
        medoid_idx_i = medoids_idx[i]
        min_dist = np.inf

        for j in range(M):
            if i != j:
                medoid_idx_j = medoids_idx[j]
                d = distance_metrics.get_distance(X[medoid_idx_i], X[medoid_idx_j], metric)

                if d < min_dist:
                    min_dist = d

        radius[i] = min_dist/2

    return radius

@cython.boundscheck(False)
@cython.wraparound(False)
cdef cnp.int64_t[:] set_diff(cnp.int64_t[:] arr1, cnp.int64_t[:] arr2):    
    return np.setdiff1d(arr1, arr2, True)
