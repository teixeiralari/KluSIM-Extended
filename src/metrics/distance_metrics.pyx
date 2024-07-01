import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport sqrt
from cython cimport floating

cnp.import_array()
# Declare the distance functions
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double euclidean_distance(floating[:] a, floating[:] b):
    cdef floating dist = 0.0
    cdef Py_ssize_t dim = len(a)
    cdef floating diff

    for i in range(dim):
        diff = (a[i] - b[i])
        dist += diff * diff
    return sqrt(dist)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double cosine_distance(floating[:] a, floating[:] b):
    cdef floating dot_product = 0.0
    cdef floating magnitude_a = 0.0
    cdef floating magnitude_b = 0.0
    cdef Py_ssize_t i
    cdef Py_ssize_t dim = len(a)

    for i in range(dim):
        dot_product += a[i] * b[i]
        magnitude_a += a[i] * a[i]
        magnitude_b += b[i] * b[i]

    magnitude_a = sqrt(magnitude_a)
    magnitude_b = sqrt(magnitude_b)

    dist = 1 - (dot_product / (magnitude_a * magnitude_b))

    return dist

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double manhattan_distance(floating[:] a, floating[:] b):
    cdef floating dist = 0.0
    cdef Py_ssize_t i
    cdef Py_ssize_t dim = len(a)

    for i in range(dim):
        dist += abs(a[i] - b[i])

    return dist

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double chebyshev_distance(floating[:] a, floating[:] b):
    cdef floating max_difference = 0.0
    cdef Py_ssize_t i
    cdef Py_ssize_t dim = len(a)

    for i in range(dim):
        diff = abs(a[i] - b[i])

        if diff > max_difference:
            max_difference = diff

    return max_difference

cdef double get_distance(floating[:] a, floating[:] b, str metric):
    if metric == 'euclidean':
        return euclidean_distance(a, b)
    elif metric == 'manhattan':
        return manhattan_distance(a, b)
    elif metric == 'chebyshev':
        return chebyshev_distance(a, b)