cimport cython
import numpy as np
cimport numpy as cnp
from src.vptree cimport vptree
from src.vptree import vptree
from sklearn.neighbors import KDTree, BallTree
from src.metrics import distance_metrics
from src.metrics cimport distance_metrics
from src.utils cimport helpers
from src.utils import helpers

cnp.import_array()

cdef class KluSIM():
    cdef:
        int n_clusters 
        str metric
        str reference_point
        cnp.uint64_t max_iter 
        public double[:, :] cluster_centers_
        public cnp.int64_t[:] medoid_indices_
        public cnp.int64_t N
        public double inertia_
        public object data
        public tree
        bint medoids_is_set
        public cnp.int64_t[:] init_medoids_idx
        public cnp.uint64_t number_of_calc_dist
        public cnp.uint64_t number_of_calc_dist_build
        str access_method_tree
        cnp.uint64_t number_of_calc_dist_tree
        bint keepdims

    def __init__(self, int n_clusters, str ref_point='mean', str metric='euclidean', str access_method_tree='KDTree', cnp.int32_t max_iter=1000):
        self.n_clusters = n_clusters
        self.metric = metric
        self.max_iter = max_iter
        self.reference_point = ref_point
        self.access_method_tree = access_method_tree

    cdef build(self, double[:,:] X):
        if self.access_method_tree == 'KDTree':
            self.tree = KDTree(X, metric=self.metric)
            self.keepdims = True
        elif self.access_method_tree == 'BallTree':
            self.tree = BallTree(X, metric=self.metric)
            self.keepdims = True
        elif self.access_method_tree == 'VPTree':
            self.tree = vptree.VPTree(X, self.metric)
            self.keepdims = False
        else:
            raise Exception('Method Access not implemented.')
        self.number_of_calc_dist_build = self.tree.get_n_calls()

    def build_tree(self, X):
        self.build(X)

    def set_medoids_idx(self, cnp.int64_t[:] medoids):
        self.init_medoids_idx = medoids
        self.medoids_is_set = True

    def swap(self, double[:,:] X, int p=3):
        self.N = X.shape[0]
        self.number_of_calc_dist = 0

        cdef cnp.int64_t[:] medoids_idxs = np.copy(self.init_medoids_idx) 
  
        dataset_idx = np.arange(self.N, dtype=np.int64)
        
        self.build_tree(X)

        total_deviation, cluster = self.assurance_similarity_query(X, medoids_idxs, dataset_idx)

        for _ in range(self.max_iter):
            swap = self._compute_optimal_swap(
                    X,
                    medoids_idxs,
                    cluster,
                    dataset_idx,
                    total_deviation,
                    self.n_clusters,
                    p
                )

            self.number_of_calc_dist_tree += self.tree.get_n_calls()
            self.tree.reset_n_calls()

            if swap:
                total_deviation = swap[0]
                medoids_idxs = swap[1]
                cluster = swap[2]
            else:
                break
        
        self.cluster_centers_ = np.array(X)[medoids_idxs]
        self.medoid_indices_ = medoids_idxs
        self.inertia_ = total_deviation
        self.number_of_calc_dist += self.number_of_calc_dist_tree
        self.data = cluster
        return self
    
    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)
    cdef _compute_optimal_swap(
            self, 
            double[:,:] X, 
            cnp.int64_t[:] medoids_idxs, 
            list cluster_idx, 
            cnp.int64_t[:] dataset_idx,  
            double total_deviation, 
            int k, 
            int p
        ):

        cdef double best_cost = total_deviation
        cdef list cluster = cluster_idx
        cdef cnp.int64_t[:] best_swaps_medoids = medoids_idxs.copy()
        cdef int m

        for m in range(k):
            id_i = medoids_idxs[m]

            if self.reference_point == 'mean':
                u_i = np.mean(X.base[cluster_idx[m]], axis=0, keepdims=self.keepdims)
            else:
                u_i = np.median(X.base[cluster_idx[m]], axis=0, keepdims=self.keepdims)

            S_p = self.tree.query(u_i, p, return_distance=False)[0]
        
            for o_j in S_p:
    
                medoids_idxs[m] = o_j

                new_cost, new_cluster = self.assurance_similarity_query(X, medoids_idxs, dataset_idx)

                if new_cost < best_cost:
                    best_cost = new_cost
                    best_swaps_medoids[m] = o_j
                    cluster = new_cluster
                else:
                    medoids_idxs[m] = id_i

        if best_cost < total_deviation:
            return best_cost, best_swaps_medoids, cluster
        else:
            return None

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)
    cdef assurance_similarity_query(self, double[:,:] X, cnp.int64_t[:] s_q, cnp.int64_t[:] X_idx):

        cdef double[:] radius = helpers.find_radius(X, s_q, self.metric)
        cdef list clusters = [None] * self.n_clusters
        cdef double minimum_distances = 0.0
        cdef list all_idx = []
        cdef Py_ssize_t obj_idx, medoids_idx, idx
        cdef double dist = 0.0
        cdef double min_dist = np.inf
        cdef Py_ssize_t min_medoid_dist = -99

        for idx in range(self.n_clusters):
    
            objects_covered_idx, distances = self.tree.query_radius([X[s_q[idx]]], radius[idx], return_distance=True)
           
            all_idx.extend(objects_covered_idx[0])
            if self.access_method_tree == 'VPTree':
                clusters[idx] = objects_covered_idx[0]
            else:
                clusters[idx] = objects_covered_idx[0].tolist()

            minimum_distances += np.sum(distances[0])
 
        cdef cnp.int64_t[:] all_idx_np = np.asarray(all_idx, dtype=np.int64)
        cdef cnp.int64_t[:] objs_not_covered = helpers.set_diff(X_idx, all_idx_np)

        cdef cnp.int64_t objs_not_covered_shape = objs_not_covered.shape[0]

        if objs_not_covered_shape > 0:

            for obj_idx in range(objs_not_covered_shape):
                min_dist = np.inf

                for medoids_idx in range(self.n_clusters):
                    dist = distance_metrics.get_distance(X[objs_not_covered[obj_idx]], X[s_q[medoids_idx]], self.metric)
                    self.number_of_calc_dist += 1

                    if dist < min_dist:
                        min_medoid_dist = medoids_idx
                        min_dist = dist
                
                minimum_distances += min_dist
                clusters[min_medoid_dist].append(objs_not_covered[obj_idx])
        
        return minimum_distances, clusters

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