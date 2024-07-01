import numpy as np
cimport numpy as cnp
from cython cimport floating
# Declare the distance functions
cdef double euclidean_distance(floating[:] a, floating[:] b)
cdef double cosine_distance(floating[:] a, floating[:] b)
cdef double manhattan_distance(floating[:] a, floating[:] b)
cdef double chebyshev_distance(floating[:] a, floating[:] b)
cdef double get_distance(floating[:] a, floating[:] b, str metric)