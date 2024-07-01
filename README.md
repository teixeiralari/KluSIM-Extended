# KluSIM: <u>k</u>-medoids Cl<u>u</u>stering <u>S</u>wap <u>I</u>mprovement with Access <u>M</u>ethod

This extended work builds upon [KluSIM](https://github.com/teixeiralari/KluSIM) with the following enhancements:

- **Distance Metrics**: Added options to run the algorithm using various Minkowski family distances, including Manhattan, Euclidean, and Chebyshev.
- **Access Methods**: Incorporated additional access methods such as VPTree, KDTree, and BallTree.
- **Medoid Selection**: Introduced the option to use either the centroid or median point for selecting medoid candidates.
- **Algorithm Expansion**: Included more state-of-the-art k-medoids algorithms.

## Installation

Make sure you have Python 3 installed. Then, execute the following commands:

```bash
pip3 install -r requirements.txt
python3 setup.py build_ext --inplace
```

## Usage
### Input Parameters
The algorithm takes the following input parameters:

- X: The input dataset to be clustered.
- k: The number of clusters to form.
- medoids: Indices of the initial medoids.

### Usage Example
To run the *KluSIM* algorithm, follow these steps:

1. Given a dataset, select the initial *k* medoids using *BUILD* or *k-means++* heuristics.

    ```python

    import kmedoids_initialization
    from sklearn.datasets import make_blobs
    import time
    import numpy as np

    n_samples = 3000
    n_features = 8
    n_clusters = 5
    random_state = 102

    X, _ = make_blobs(n_samples=n_samples, n_features=n_features, 
                        centers=n_clusters, random_state=random_state)

    heuristic = 'BUILD' # e.g. 'BUILD' or 'k-means++'
    metric = 'euclidean' # e.g. 'euclidean' or 'manhattan' or 'chebyshev'.

    medoids = kmedoids_initialization.InitializeMedoids(X, n_clusters, heuristic=heuristic, metric=metric)

    ```
*Note*: The k-means++ was created with the goal of accelerating the convergence of the k-means algorithm. Therefore, it is possible to test the k-means++ as an initialization method only for Euclidean distance.


2. Then, set the medoids, and call swap method:

    ```python
    ks = klusim.KluSIM(n_clusters, metric=metric, access_method_tree='KDTree', ref_point = 'mean') # ref_point can be either 'mean' or 'median'
    ks.set_medoids_idx(medoids)
    ks_results = ks.swap(X)

    print("Average Distance - KluSIM: %.3f" % ks_results.average_distance(X, ks_results.medoid_indices_)) # The lower the better
    print("#distance calculations - KluSIM: %.3f" % ks_results.number_of_calc_dist)

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
    ```

### Usage Example Script

We have provided a script (*usage_example.py*) on how to use the *KluSIM* algorithm with a sample dataset. To run it, execute the following command:

```bash
    python3 usage_example.py
```