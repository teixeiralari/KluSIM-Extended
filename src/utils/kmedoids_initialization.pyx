cimport cython
import numpy as np
cimport numpy as cnp
from src.metrics import distance_metrics
from src.metrics cimport distance_metrics
from sklearn.cluster import kmeans_plusplus
from cython cimport floating
from sklearn.metrics.pairwise import (
    pairwise_distances
)
    
def _build( floating[:, :] X, int n_clusters, str metric):
    """Compute BUILD initialization, a greedy medoid initialization."""

    cdef floating[:, :] D = pairwise_distances(X, metric=metric)

    cdef int[:] medoid_idxs = np.zeros(n_clusters, dtype = np.intc)
    cdef int sample_size = len(D)
    cdef int[:] not_medoid_idxs = np.arange(sample_size, dtype = np.intc)
    cdef int i, j,  id_i, id_j

    medoid_idxs[0] = np.argmin(np.sum(D,axis=0))
    not_medoid_idxs = np.delete(not_medoid_idxs, medoid_idxs[0])

    cdef int n_medoids_current = 1

    cdef floating[:] Dj = D[medoid_idxs[0]].copy()
    cdef floating cost_change
    cdef (int, int) new_medoid = (0,0)
    cdef floating cost_change_max

    for _ in range(n_clusters -1):
        cost_change_max = 0
        for i in range(sample_size - n_medoids_current):
            id_i = not_medoid_idxs[i]
            cost_change = 0
            for j in range(sample_size - n_medoids_current):
                id_j = not_medoid_idxs[j]
                cost_change +=   max(0, Dj[id_j] - D[id_i, id_j])
            if cost_change >= cost_change_max:
                cost_change_max = cost_change
                new_medoid = (id_i, i)


        medoid_idxs[n_medoids_current] = new_medoid[0]
        n_medoids_current +=  1
        not_medoid_idxs = np.delete(not_medoid_idxs, new_medoid[1])


        for id_j in range(sample_size):
            Dj[id_j] = min(Dj[id_j], D[id_j, new_medoid[0]])
            
    return np.array(medoid_idxs)

cdef _kmeans_plusplus(X, n_clusters):
    """
    This is a scikit-learn package:
        https://scikit-learn.org/stable/modules/generated/sklearn.cluster.kmeans_plusplus.html#sklearn.cluster.kmeans_plusplus
    """

    X = np.asarray(X)
    _, medoids_idx = kmeans_plusplus(X, n_clusters)
    return np.array(medoids_idx)

def InitializeMedoids(floating[:, :] X, int n_clusters, str heuristic='BUILD', str metric='euclidean'):
    if heuristic == 'BUILD':
        return _build(X, n_clusters, metric).astype(dtype=np.int64)
    elif heuristic=='k-means++':
        return _kmeans_plusplus(X, n_clusters).astype(dtype=np.int64)
    else:
        raise Exception("Method not implemented. Please select 'BUILD' or 'k-means++'")