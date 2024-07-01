cimport cython
cimport numpy as cnp
import numpy as np
from src.metrics import distance_metrics
from src.metrics cimport distance_metrics

cnp.import_array()

cdef class VPTreeNode:
    def __init__(self, double[:] pivot, double threshold, int index=-1, VPTreeNode left=None, VPTreeNode right=None):
        self.pivot = pivot
        self.left = left
        self.right = right
        self.index = index
        self.threshold = threshold 
        
cdef class VPTree:
    def __init__(self, points, metric='euclidean'):
        self.number_of_calc_dist = 0
        self.metric = metric
        self.root = self.build(points, np.arange(points.shape[0], dtype=np.int64))
        

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef VPTreeNode build(self, double[:,:] points, cnp.int64_t[:] index):
        
        cdef int x_max = points.shape[0] - 1
        cdef int y_max = points.shape[1]
        
        if points.shape[0] == 0:
            return None

        # print(np.asarray(points))
        cdef int pivot_index = index[0]
        cdef double[:] pivot = points[0]

        cdef cnp.int64_t[:] points_index = index[1:]

        points = points[1:]

        cdef cnp.ndarray distances = np.array([distance_metrics.get_distance(points[iterator], pivot, self.metric) for iterator in range(points.shape[0])])
        self.number_of_calc_dist += points.shape[0]

        cdef double threshold = np.median(distances) if distances.shape[0] > 0 else 0
        cdef cnp.int64_t[:] right_index = np.zeros((x_max, ), dtype=np.int64) 
        cdef cnp.int64_t[:] left_index = np.zeros((x_max, ), dtype=np.int64)

        cdef double[:, :] left_points = np.zeros((x_max, y_max), dtype=np.float64)
        cdef double[:, :] right_points = np.zeros((x_max, y_max), dtype=np.float64)

        cdef int count_idx_right = 0
        cdef int count_idx_left = 0

        for row in range(x_max):
            if distances[row] < threshold:
                left_points[count_idx_left] = points[row]
                left_index[count_idx_left] = points_index[row]
                count_idx_left += 1
            else:
                right_points[count_idx_right] = points[row]
                right_index[count_idx_right] = points_index[row]
                count_idx_right += 1
                
        cdef VPTreeNode node = VPTreeNode(pivot, threshold, pivot_index)
  
        if count_idx_left > 0:
            node.left = self.build(left_points[:count_idx_left], left_index[:count_idx_left])

        if count_idx_right > 0:
            node.right = self.build(right_points[:count_idx_right], right_index[:count_idx_right])

        return node

    def query(self, target, k, return_distance=True): #knn query
        knn = []
        self._knn_search(self.root, knn, target, k)

        indices = [r[0] for r in knn]
        distances = [r[1] for r in knn]

        if return_distance:
            results = [indices, distances]
        else:
            results = [indices]

        return results
    
    # cdef _knn_search(self, VPTreeNode node, list results, cnp.float64_t[:] target, int p):
    cdef _knn_search(self, VPTreeNode node, list results, double[:] target, int k):
        if node is None:
            return

        distance = distance_metrics.get_distance(target, node.pivot, self.metric)
        self.number_of_calc_dist += 1

        if len(results) < k:
            results.append([node.index, distance])
            results.sort(key=lambda x: x[1], reverse=False)

        elif distance < results[-1][1]:
            results[-1] = [node.index, distance]
            results.sort(key=lambda x: x[1], reverse=False)

        if distance - node.threshold < results[-1][1]:
            self._knn_search(node.left, results, target, k)

        if distance + results[-1][1] >= node.threshold:
            self._knn_search(node.right, results, target, k)

    def query_radius(self, list query_point, double radius, return_distance=False):
        distances, objs = [], []
        self._range_query_recursive(self.root, query_point[0], radius, objs, distances)
        
        if return_distance:
            results = [[objs], [distances]]
        else:
            results = [objs]

        return results
    
    cdef _range_query_recursive(self, VPTreeNode node, double[:] query_point, 
                                double radius, list objs, list distances):

        if node is None:
            return 
        
        cdef double distance = distance_metrics.get_distance(query_point, node.pivot, self.metric)
        self.number_of_calc_dist += 1
       
        if distance <= radius:
            objs.append(node.index)
            distances.append(distance)
          
        if distance - node.threshold <= radius:
            self._range_query_recursive(node.left, query_point, radius, objs, distances)

        if node.threshold - distance <= radius:
            self._range_query_recursive(node.right, query_point, radius, objs, distances)
    
    def get_n_calls(self):
        return self.number_of_calc_dist

    def reset_n_calls(self):
        self.number_of_calc_dist = 0