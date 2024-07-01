import kmedoids_initialization
import fasterpam 
import klusim
import sfkm
import fames
from sklearn.datasets import make_blobs


n_samples = 3000
n_features = 8
n_clusters = 5
random_state = 102

X, _ = make_blobs(n_samples=n_samples, n_features=n_features, 
                    centers=n_clusters, random_state=random_state)

heuristic = 'BUILD' # e.g. 'BUILD' or 'k-means++'
metric = 'euclidean' # e.g. 'euclidean' or 'manhattan' or 'chebyshev'.
medoids = kmedoids_initialization.InitializeMedoids(X, n_clusters, heuristic=heuristic, metric=metric)

# Method BUILD
medoids = kmedoids_initialization.InitializeMedoids(X, n_clusters, heuristic='BUILD', metric='euclidean')
ks = klusim.KluSIM(n_clusters, metric=metric, access_method_tree='KDTree', ref_point = 'mean')
ks.set_medoids_idx(medoids)
ks_results = ks.swap(X)

print("Average Distance - KluSIM with 'mean' as reference point: %.3f" % ks_results.average_distance(X, ks_results.medoid_indices_)) # The lower the better
print("#distance calculations - KluSIM with 'mean' as reference point: %.3f" % ks_results.number_of_calc_dist)

medoids = kmedoids_initialization.InitializeMedoids(X, n_clusters, heuristic='BUILD', metric=metric)
ks = klusim.KluSIM(n_clusters, metric=metric, access_method_tree='KDTree', ref_point = 'median')
ks.set_medoids_idx(medoids)
ks_results = ks.swap(X)

print("Average Distance - KluSIM with 'median' as reference point: %.3f" % ks_results.average_distance(X, ks_results.medoid_indices_)) # The lower the better
print("#distance calculations - KluSIM with 'median' as reference point: %.3f" % ks_results.number_of_calc_dist)


# # # Method BUILD
fp = fasterpam.FasterPAM(n_clusters, metric=metric)
fp.set_medoids_idx(medoids)
fp_results = fp.swap(X)

print("Average Distance - FasterPAM: %.3f" % fp_results.average_distance(X, fp_results.medoid_indices_)) # The lower the better
print("#distance calculations - FasterPAM: %.3f" % fp_results.number_of_calc_dist)

fames_ = fames.FAMES(n_clusters, metric=metric)
fames_.set_medoids_idx(medoids)
fames_results = fames_.swap(X)

print("Average Distance - FAMES: %.3f" % fames_results.average_distance(X, fames_results.medoid_indices_)) # The lower the better
print("#distance calculations - FAMES: %.3f" % fames_results.number_of_calc_dist)


sfkm_ = sfkm.SFKM(n_clusters, metric=metric)
sfkm_.set_medoids_idx(medoids)
sfkm_results = sfkm_.swap(X)

print("Average Distance - SFKM: %.3f" % sfkm_results.average_distance(X, sfkm_results.medoid_indices_)) # The lower the better
print("#distance calculations - SFKM: %.3f" % sfkm_results.number_of_calc_dist)
