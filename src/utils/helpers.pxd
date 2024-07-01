cimport numpy as cnp

cdef cnp.int64_t[:] set_diff(cnp.int64_t[:]arr1, cnp.int64_t[:] arr2)
cdef double[:] find_radius(double[:,:] X, cnp.int64_t[:] medoids_idx, str metric)