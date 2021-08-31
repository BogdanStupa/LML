import numpy as np
import copy

from mylearn.cluster.base_cluster import BaseCluster
from mylearn.cluster.kmeans._kmeans_common import _clustering, _inertia, _tolerance, _recalculate_centers
from mylearn.utilities.plot_data_sample.make_gif_2d_clustered_sample import make_gif_2d_clusterd_sample


def kmeans_single(
        X,
        centers,
        max_iter=100,
        verbose=False,
        make_gif=False,
        tolerance=1e-4
):
    labels = np.zeros(shape=(X.shape[0]), dtype=int)
    _clustering(X, labels, centers)

    if make_gif:
        data_for_gif = [[labels, centers]]

    for i in range(max_iter):
        prev_centers = copy.deepcopy(centers)
        _recalculate_centers(X, labels, centers)
        _clustering(X, labels, centers)

        if make_gif:
            data_for_gif.append([copy.deepcopy(labels), copy.deepcopy(centers)])
        if _tolerance(centers, prev_centers, tolerance):
            break

    sample_weight = []
    inertia = _inertia(X, sample_weight, centers, labels)

    if make_gif:
        make_gif_2d_clusterd_sample(X, centers.shape[0], data_for_gif)

    return labels, centers, inertia, i + 1


class MyKMeans(BaseCluster):
    __slots__ = ["n_clusters", "verbose", "max_iter", "n_init", "accuracy", "copy_x", "make_gif", "labels_"]

    def __init__(
            self,
            n_clusters=3,
            init="k-means",
            max_iter=7,
            n_init=1,
            verbose=False,
            accuracy=0.002,
            copy_x=True,
            make_gif=False
    ):
        self.n_clusters = n_clusters
        self.verbose = verbose
        self.max_iter = max_iter
        self.n_init = n_init
        self.init = init
        self.accuracy = accuracy
        self.copy_x = copy_x
        self.make_gif = make_gif
        self.labels_ = None

    def _check_params(self, X):
        pass

    def fit(self, X):
        self._check_params(X)

        # _normalize(X)
        centers = None
        labels = None

        for i in range(self.n_init):
            centers = self._init_centers(X)
            if self.verbose:
                print("Initialization complete")
            labels, centers, inertia, n_iter = kmeans_single(
                X,
                centers,
                self.max_iter,
                make_gif=self.make_gif
            )
            # select best inertia
        self.centers = centers
        self.labels_ = labels
        return self

    def predict(self, X):
        # _normalize(X)
        labels = np.zeros(shape=(X.shape[0]), dtype=int)
        _clustering(X, labels, self.centers)
        return labels

    def _init_centers(self, X):
        set_of_idx = set()
        k = 1
        centers = np.ndarray(shape=(self.n_clusters, X.shape[-1]))
        while k <= self.n_clusters:
            idx = np.random.randint(X.shape[0])
            if idx not in set_of_idx:
                set_of_idx.add(idx)
                centers[k - 1] = X[idx]
                k += 1
        return centers

    def get_centers(self):
        return self.centers

    def get_labels(self):
        return self.labels_