cimport cython
import numpy as np
cimport numpy as cnp
from src.metrics import distance_metrics
from src.metrics cimport distance_metrics
from cython cimport double

cnp.import_array()

cdef class Data:
    cdef public double nearest
    cdef public double second
    cdef public double distance_nearest
    cdef public double distance_second_nearest

    def __init__(self, double nearest=np.inf, double second=np.inf, double distance_nearest=np.inf, double distance_second_nearest=np.inf):
        self.nearest = nearest
        self.second = second
        self.distance_nearest = distance_nearest
        self.distance_second_nearest = distance_second_nearest

    def get_nearest_idx(self):
        return int(self.nearest)

    def get_nearest_distance(self):
        return self.distance_nearest

    def get_second_nearest_idx(self):
        return int(self.second)
    
    def get_second_nearest_distance(self):
        return self.distance_second_nearest

    def set_nearest(self, double i, double d):
        self.nearest = i
        self.distance_nearest = d

    def set_second_nearest(self, double i, double d):
        self.second = i
        self.distance_second_nearest = d

cdef class FasterPAM():
    """
    The cython version of FasterPAM was implemented based on the original Rust code available at
       https://github.com/kno10/rust-kmedoids/blob/main/src/fasterpam.rs
    """

    cdef int n_clusters 
    cdef str metric
    cdef cnp.uint64_t max_iter 
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

    def swap(self, double[:,:] X):
        self.number_of_calc_dist = 0
 
        cdef cnp.int64_t[:] medoids_idxs = np.copy(self.init_medoids_idx) 
        loss, data = self.initial_assigment(X, medoids_idxs)
        
        cdef double[:] removal_loss = self.update_removal_loss(data)
        cdef Py_ssize_t x_last = -99
        cdef Py_ssize_t n_swap = 0
        cdef Py_ssize_t iter, x_c, swaps_before

        for _ in range(self.max_iter):
            swaps_before = n_swap

            for x_c in range(X.shape[0]):

                if x_c == x_last:
                    break

                if x_c == medoids_idxs[data[x_c].get_nearest_idx()]:
                    continue

                i, ploss = self.find_best_swap(removal_loss, X, x_c, data)

                if ploss >= 0:
                    continue

                x_last = x_c
                n_swap += 1


                newloss, medoids_idxs, data = self.do_swap(
                    X, medoids_idxs, data, i, x_c)

                if newloss >= loss:
                    break

                loss = newloss
                removal_loss = self.update_removal_loss(data)
   
            if n_swap == swaps_before:
                break

        self.cluster_centers_ = np.array(X)[medoids_idxs]
        self.medoid_indices_ = np.array(medoids_idxs)
        self.inertia_ = loss
        self.data = data
        return self

    cdef initial_assigment(self, double[:,:] X, cnp.int64_t[:] medoids_idx):

        cdef cnp.ndarray data = cnp.ndarray((X.shape[0],), dtype=Data)
        cdef cnp.int64_t first_medoid = medoids_idx[0]

        cdef double loss = 0.0
        cdef cnp.int64_t o, m, medoid_idx

        for o in range(X.shape[0]):
            first_d = distance_metrics.get_distance(X[o], X[first_medoid], self.metric)
            self.number_of_calc_dist += 1
            
            obj = Data(nearest=0, distance_nearest=first_d)
            
            for m in range(1, medoids_idx.shape[0]):
              
                medoid_idx = medoids_idx[m]
                d = distance_metrics.get_distance(X[o], X[medoid_idx], self.metric)
                self.number_of_calc_dist += 1

                if (d < obj.distance_nearest) or (o == medoid_idx):
                    obj.second = obj.nearest
                    obj.distance_second_nearest = obj.distance_nearest
                    obj.nearest = m
                    obj.distance_nearest = d
                elif (obj.distance_second_nearest == np.inf) or (d < obj.distance_second_nearest):
                    obj.second = m
                    obj.distance_second_nearest = d
                    
            loss += obj.distance_nearest
            data[o] = obj
        return loss, data

    cdef do_swap(self, double[:,:] X, cnp.int64_t[:] medoids_idx, cnp.ndarray data, cnp.int32_t b, cnp.int32_t j):

        medoids_idx[b] = j
        cdef double loss = 0.0
        cdef cnp.int64_t o

        for o in range(data.shape[0]):

            if o == j:
                if data[o].get_nearest_idx() != b:
                    data[o].set_second_nearest(data[o].get_nearest_idx(), data[o].get_nearest_distance())
                   
                data[o].set_nearest(b, 0.0)
            
            djo = distance_metrics.get_distance(X[j], X[o], self.metric) #X[j, o]
            self.number_of_calc_dist += 1
            
            # Nearest medoid is gone:
            if data[o].get_nearest_idx() == b:
                if djo < data[o].get_second_nearest_distance():
                    data[o].set_nearest(b, djo)

                else:
                    data[o].set_nearest(data[o].get_second_nearest_idx(), data[o].get_second_nearest_distance())

                    i, d = self.update_second_nearest(
                        X, medoids_idx, data[o].get_nearest_idx(), b, o, djo)
                    
                    data[o].set_second_nearest(i, d)
            else:
                if djo < data[o].get_nearest_distance():
                    data[o].set_second_nearest(data[o].get_nearest_idx(), data[o].get_nearest_distance())
                    data[o].set_nearest(b, djo)

                
                elif djo < data[o].get_second_nearest_distance():
                    data[o].set_second_nearest(b, djo)

                    
                elif data[o].get_second_nearest_idx() == b:
                    
                    i, d = self.update_second_nearest(
                        X, medoids_idx, data[o].get_nearest_idx(), b, o, djo)
                    
                    data[o].set_second_nearest(i, d)
                   
            loss += data[o].get_nearest_distance()

        return loss, medoids_idx, data

    cdef update_second_nearest(self, double[:,:] X, cnp.int64_t[:] medoids_idx, cnp.int32_t nearest, cnp.int32_t b, cnp.int32_t o, double djo):
        cdef int m
        cdef double d = 0.0

        second_idx, second_distance = (b, djo)
        
        for m in range(self.n_clusters):

            if m == nearest or m == b:
                continue

            d = distance_metrics.get_distance(X[o], X[medoids_idx[m]], self.metric) #X[o, medoids_idx[m]]
            self.number_of_calc_dist += 1

            if d < second_distance:
                second_idx, second_distance = (m, d)

        return second_idx, second_distance

    cdef double[:] update_removal_loss(self, cnp.ndarray data):
        cdef double[:] loss = np.zeros(self.n_clusters, dtype=np.float64)
        cdef Py_ssize_t o

        for o in range(data.shape[0]):
            
            loss[data[o].get_nearest_idx()] += data[o].get_second_nearest_distance() - \
                data[o].get_nearest_distance()

        return loss

    cdef find_best_swap(self, double[:] loss, double[:,:] X, cnp.int32_t j, cnp.ndarray data):
        cdef double[:] ploss = np.copy(loss)
        cdef double delta_xc = 0.0
        cdef Py_ssize_t o, i
        cdef double d, bloss
        
        for o in range(data.shape[0]):
            d = distance_metrics.get_distance(X[j], X[o], self.metric) #X[j, o]
            self.number_of_calc_dist += 1

            if d < data[o].get_nearest_distance():
                delta_xc += d - data[o].get_nearest_distance()
                ploss[data[o].get_nearest_idx()] += data[o].get_nearest_distance() - \
                    data[o].get_second_nearest_distance()

            elif d < data[o].get_second_nearest_distance():
                ploss[data[o].get_nearest_idx()] += d - data[o].get_second_nearest_distance()

        i = np.argmin(ploss)
        bloss = ploss[i] + delta_xc
        return i, bloss

    cdef pam_build(self, double[:,:] X):
        cdef int N = X.shape[0]
        cdef int i, j, m
        cdef (double, int) best = (0.0, self.n_clusters)
        cdef double sum_, loss
        cdef cnp.int64_t[:] medoids_idx = np.empty((self.n_clusters,), dtype=np.int64)
        cdef cnp.ndarray data = cnp.ndarray((N,), dtype=Data)

        for i in range(N):
            sum_ = 0.0
            for j in range(N):
                if i != j:
                    sum_ += X[i, j]
            
            if (i == 0) or (sum_ < best[0]):
                best = (sum_, i)


        loss = best[0]
        medoids_idx[0] = best[1]

        for j in range(N):
            data[j] = Data(nearest=0, distance_nearest=X[best[1], j])

        for m in range(1, self.n_clusters):
            best = (0.0, self.n_clusters)
            for o in range(N):
                sum_ = - data[o].get_nearest_distance()
                for j in range(N):
                    if j != o:
                        d = X[i, j]
                        if d < data[j].get_nearest_distance():
                            sum_ += d - data[j].get_nearest_distance()
                if (o == 0) or (sum_ < best[0]):
                    best = (sum_, o)

            if best[0] > 0.0:
                break
            
            loss = 0.0
            for j in range(N):
                if j == best[1]:
                    data[j].set_second_nearest(data[j].get_nearest_idx(), data[j].get_nearest_distance())
                    data[j].set_nearest(m, 0)
                    continue
                
                dj = X[best[1], j]
                if dj < data[j].get_nearest_distance():
                    data[j].set_second_nearest(data[j].get_nearest_idx(), data[j].get_nearest_distance())
                    data[j].set_nearest(m, dj)
                elif (data[j].second == np.inf) or (dj < data[j].get_second_nearest_distance()):
                    data[j].set_second_nearest(m, dj)
                
                loss += data[j].get_nearest_distance()
            
            medoids_idx[m] = best[1]

        
        return loss, medoids_idx, data

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