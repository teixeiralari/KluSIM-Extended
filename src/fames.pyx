cimport cython
import numpy as np
cimport numpy as cnp
from src.metrics import distance_metrics
from src.metrics cimport distance_metrics

cdef class FAMES:
    cdef:
        int n_clusters 
        str metric
        cnp.uint64_t max_iter 
        public double[:, :] cluster_centers_
        public cnp.int64_t[:] medoid_indices_
        public cnp.int64_t N
        public double inertia_
        public object data
        bint medoids_is_set
        public cnp.int64_t[:] init_medoids_idx
        public cnp.uint64_t number_of_calc_dist

    def __init__(self, int n_clusters, str metric='euclidean', cnp.int32_t max_iter=1000):
        self.n_clusters = n_clusters
        self.metric = metric
        self.max_iter = max_iter

    cdef assign_nearest(self, double[:,:] X, cnp.int64_t[:] medoids_idx):
        cdef Py_ssize_t N = X.shape[0]
        cdef list clusters = [[] for _ in range(self.n_clusters)]
        cdef double loss = 0.0
        cdef Py_ssize_t o, m
        cdef double best_d = np.inf
        cdef Py_ssize_t best_m

        for o in range(N):
            best_m = -1
            best_d = np.inf

            for m in range(self.n_clusters):
                m_idx = medoids_idx[m]
                distance = distance_metrics.get_distance(X[o], X[m_idx], self.metric)
                self.number_of_calc_dist += 1

                if distance < best_d:
                    best_d = distance
                    best_m = m

            clusters[best_m].append(o)
            loss += best_d
        return clusters, loss

    def set_medoids_idx(self, cnp.int64_t[:] medoids):
        self.init_medoids_idx = medoids
        self.medoids_is_set = True
    
    def swap(self, double[:,:] X):
        cdef int m
        cdef double loss
        cdef list clusters

        self.N = X.shape[0]

        self.number_of_calc_dist = 0

        cdef cnp.int64_t[:] medoids_idx = np.copy(self.init_medoids_idx)
        clusters, loss = self.assign_nearest(X, self.init_medoids_idx)
        
        for _ in range(self.max_iter):
            old_medoids = np.copy(medoids_idx)
            
            for m in range(self.n_clusters):
                if len(clusters[m]) > 1:
                    distance_pivots, distance_s1_s2 = self.find_pivots(X, clusters[m])
                    x_md = self.projection_over_line_s1_s2(distance_pivots, distance_s1_s2)
                    best_medoid = self.choose_medoid(x_md, distance_pivots, distance_s1_s2)
                else:
                    best_medoid = 0
                medoids_idx[m] = clusters[m][best_medoid]

            clusters, loss = self.assign_nearest(X, medoids_idx)
            if np.all(medoids_idx == old_medoids):
                break

        self.inertia_ = loss
        self.medoid_indices_ = medoids_idx
        self.data = clusters
        self.cluster_centers_ = np.array(X)[medoids_idx]
        return self


    cdef find_pivots(self, double[:,:] X, list cluster):
        cdef Py_ssize_t N = len(cluster)
        cdef double[:,:] distance_pivots = np.zeros((N,2))
        cdef Py_ssize_t sz = np.random.choice(N, 1, replace=False)[0]
        cdef double d_farthest_sz = 0.0
        cdef Py_ssize_t s1 = -1
        cdef double d_farthest_s1 = 0.0
        cdef Py_ssize_t s2 = -1
        cdef Py_ssize_t i
        cdef double d
        
        for i in range(N): # (0..5)
            if i != sz:
                d = distance_metrics.get_distance(X[cluster[i]], X[cluster[sz]], self.metric)
                self.number_of_calc_dist += 1
                if d > d_farthest_sz:
                    s1 = i
                    d_farthest_sz = d
        
        
        for i in range(N):
            if i != s1:
                d = distance_metrics.get_distance(X[cluster[i]], X[cluster[s1]], self.metric)
                distance_pivots[i, 0] = d
                self.number_of_calc_dist += 1
                if d > d_farthest_s1:
                    s2 = i
                    d_farthest_s1 = d

        for i in range(N):
            d = distance_metrics.get_distance(X[cluster[i]], X[cluster[s2]], self.metric)
            distance_pivots[i, 1] = d
            self.number_of_calc_dist += 1

        return distance_pivots, d_farthest_s1


    cdef projection_over_line_s1_s2(self,  double[:,:] distance_pivots, double dist_s1_s2):
        cdef Py_ssize_t N = distance_pivots.shape[0]
        cdef double[:] vec_x = np.zeros((N,), dtype=np.float64)
        cdef Py_ssize_t si, median_idx
        cdef double x_md


        for si in range(N):
            vec_x[si] = ((distance_pivots[si, 0]) ** 2 + (dist_s1_s2) ** 2 - (distance_pivots[si, 1]) ** 2)/(2 * dist_s1_s2)

        cdef cnp.int64_t[:] sorted_vect_x = np.argsort(vec_x)

        if N % 2 == 0:
            median_idx =  N // 2
            x_md = (vec_x[sorted_vect_x[median_idx]] + vec_x[sorted_vect_x[median_idx-1]]) / 2
        else:
            median_idx =  N // 2
            x_md = vec_x[sorted_vect_x[median_idx]]
        
        return x_md
        
        
    cdef choose_medoid(self, double x_md, double[:,:] distance_pivots, double dist_s1_s2):
        cdef Py_ssize_t N = distance_pivots.shape[0]
        cdef Py_ssize_t best_medoid = -1
        cdef double min_dist = np.inf
        cdef Py_ssize_t o
        cdef double d

        for o in range(N):
            d = np.abs(distance_pivots[o, 0] - x_md) + np.abs(distance_pivots[o, 1] - (dist_s1_s2 - x_md))
            if d < min_dist:
                min_dist = d
                best_medoid = o

        return best_medoid

    def get_labels(self):
        cdef cnp.int64_t[:] labels = np.zeros((self.N,), dtype=np.int64)
        cdef cnp.int64_t o, m

        for m in range(self.n_clusters):
            for o in self.data[m]:
                labels[o] = m

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