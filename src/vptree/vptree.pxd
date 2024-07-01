cimport numpy as cnp
import numpy as np
from src.metrics import distance_metrics
from src.metrics cimport distance_metrics

cdef class VPTreeNode:
    cdef double[:] pivot
    cdef double threshold
    cdef int index
    cdef VPTreeNode left
    cdef VPTreeNode right

cdef class VPTree:
    cdef public cnp.uint64_t number_of_calc_dist
    cdef VPTreeNode root
    cdef str metric

    cdef VPTreeNode build(self, double[:,:] points, cnp.int64_t[:] index)
    cdef _knn_search(self, VPTreeNode node, list results, double[:] target, int k)
    
    cdef _range_query_recursive(self, VPTreeNode node, double[:] query_point, 
                                double radius, list objs, list distances)