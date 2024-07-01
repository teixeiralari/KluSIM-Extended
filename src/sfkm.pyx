cimport cython
import numpy as np
cimport numpy as cnp
from src.metrics import distance_metrics
from src.metrics cimport distance_metrics

cnp.import_array()
#
cdef class Data:
    cdef public double nearest
    cdef public double second
    cdef public double distance_nearest
    cdef public double distance_second_nearest

    def __init__(self, double nearest=np.inf, double distance_nearest=np.inf):
        self.nearest = nearest
        self.distance_nearest = distance_nearest

    def get_nearest_idx(self):
        return int(self.nearest)

    def get_nearest_distance(self):
        return self.distance_nearest

    def set_nearest(self, double i, double d):
        self.nearest = i
        self.distance_nearest = d

cdef class SFKM():
    """
    The cython version of SFKM was implemented based on the original Rust code available at
       https://github.com/kno10/rust-kmedoids/blob/main/src/alternating.rs
    """
    
    cdef cnp.int32_t n_clusters 
    cdef str metric
    cdef cnp.int32_t max_iter 
    cdef public double[:, :] cluster_centers_
    cdef public cnp.int64_t[:] medoid_indices_
    cdef public double inertia_
    cdef public cnp.ndarray data
    cdef public cnp.int64_t[:] init_medoids_idx
    cdef public cnp.int64_t number_of_calc_dist
    cdef bint medoids_is_set

    def __init__(self, cnp.int32_t n_clusters, str metric='euclidean', cnp.int32_t max_iter=1000, bint medoids_is_set = False):
        self.n_clusters = n_clusters
        self.metric = metric
        self.max_iter = max_iter
        self.medoids_is_set = medoids_is_set

    def set_medoids_idx(self, cnp.int64_t[:] medoids):
        self.init_medoids_idx = medoids
        self.medoids_is_set = True

    cdef choose_medoid(self, double[:,:] X, cnp.ndarray data, cnp.int64_t[:] medoids_idx, int m):
        cdef Py_ssize_t first = medoids_idx[m]
        cdef Py_ssize_t best = first
        cdef float sumb = 0.0
        cdef Py_ssize_t N = X.shape[0]
        cdef Py_ssize_t o, j

        for o in range(N):
            if (first != o) and (data[o].get_nearest_idx() == m):
                sumb += distance_metrics.get_distance(X[o], X[first], self.metric)
                self.number_of_calc_dist += 1

        for j in range(N):
            if (j != first) and (data[j].get_nearest_idx() == m):
                sumj = 0
                for i in range(N):
                    if (i != j) and (data[i].get_nearest_idx() == m):
                        sumj += distance_metrics.get_distance(X[i], X[j], self.metric) #mat.from_func(mat, j, i)
                        self.number_of_calc_dist += 1

                if sumj < sumb:
                    best = j
                    sumb = sumj

        medoids_idx[m] = best
        
        return best != first, medoids_idx

    def swap(self, double[:,:] X):

        self.number_of_calc_dist = 0
 
        cdef cnp.int64_t[:] medoids_idxs = np.copy(self.init_medoids_idx) 
        loss, data = self.assign_nearest(X, medoids_idxs)

        # print(loss)
        for _ in range(self.max_iter):
            changed = False

            for i in range(self.n_clusters):
                changed_, medoids_idxs = self.choose_medoid(X, data, medoids_idxs, i)
                
                if changed_:
                    changed = True
            
            if not changed:
                break

            loss, data = self.assign_nearest(X, medoids_idxs)

        self.cluster_centers_ = np.array(X)[medoids_idxs]
        self.medoid_indices_ = np.array(medoids_idxs)
        self.inertia_ = loss
        self.data = data
        return self

    cdef assign_nearest(self, double[:,:] X, cnp.int64_t[:] medoids_idx):

        cdef cnp.ndarray data = cnp.ndarray((X.shape[0],), dtype=Data)
        cdef double loss = 0.0
        cdef cnp.int64_t o, m

        for o in range(X.shape[0]):
            best_m = -1
            best_d = np.inf
            for m in range(self.n_clusters):
                m_idx = medoids_idx[m]
                distance = distance_metrics.get_distance(X[o], X[m_idx], self.metric)
                self.number_of_calc_dist += 1

                if distance < best_d:
                    best_d = distance
                    best_m = m

            data[o] = Data(nearest=best_m, distance_nearest=best_d)
            loss += best_d
            
        return loss, data

    def get_labels(self):
        cdef cnp.int64_t[:] labels = np.zeros((self.data.shape[0],), dtype=np.int64)
        cdef cnp.int64_t o

        for o in range(self.data.shape[0]):
            labels[o] = self.data[o].nearest

        return np.array(labels)

    cdef assign_objects_nearest(self, double[:,:] X, cnp.int64_t[:] medoids_idx):
        cdef Py_ssize_t N = X.shape[0]
        cdef double[:] loss = np.zeros((N,), dtype=np.float64)
        cdef Py_ssize_t o, m
        cdef double best_d = np.inf

        for o in range(N):
            best_d = np.inf

            for m in range(self.n_clusters):
                m_idx = medoids_idx[m]
                distance = distance_metrics.get_distance(X[o], X[m_idx], self.metric)
                self.number_of_calc_dist += 1

                if distance < best_d:
                    best_d = distance

            loss[o] = best_d

        return loss
        
    def average_distance(self, X, medoids_idx):
        loss = self.assign_objects_nearest(X, medoids_idx)

        return (np.sum(loss)/loss.shape[0])