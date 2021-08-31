import numpy as np
import scipy as sc
from sklearn.cluster import KMeans

from typing import Callable

from mylearn.cluster.base_cluster import BaseCluster

from ._affinity import compute_affinity
from ._kernels import squared_exponential


def laplacian(A: np.array) -> np.array:
    D = np.zeros(shape=A.shape)
    w = np.sum(A, axis=0)
    D.flat[::len(w) + 1] = w ** (-0.5)
    return D @ A @ D


def k_means(X: np.array, n_clusters: int) -> np.array:
    kmeans = KMeans(n_clusters=n_clusters).fit(X)
    return kmeans.labels_


def check_affinity(method: str) -> Callable:
    if method == "basic_affinity":
        return compute_affinity
    else:
        raise ValueError(f"{method} doesn't exist in allowed methods")


class MySpectralClustering(BaseCluster):
    __slots__ = ["n_clusters", "__cluster_method", "__sigma", "labels_", "__affinity_method", "__kernel",
                 "affinity_matrix_"]

    def __init__(
            self,
            n_clusters: int,
            affinity_method: str = "basic_affinity",
            sigma: float = .5,
            kernel: Callable[[np.array, np.array, float], np.array] = squared_exponential,
            cluster_method: Callable[[np.array, int], np.array] = k_means
    ):
        if not callable(cluster_method):
            raise TypeError("cluster method must be callabele function")
        if not callable(kernel):
            raise TypeError("kernel method must be callabele function")
        self.n_clusters = n_clusters
        self.__cluster_method = cluster_method
        self.__sigma = sigma
        self.__affinity_method = check_affinity(affinity_method)
        self.__kernel = kernel
        self.labels_ = None
        self.affinity_matrix_ = None

    def fit(self, X: np.array):
        self.affinity_matrix_ = self.__affinity_method(X, self.__kernel, self.__sigma)
        laplacian_matrix = laplacian(self.affinity_matrix_)

        eig_val, eig_vect = sc.sparse.linalg.eigs(laplacian_matrix, self.n_clusters)
        eig_vect = eig_vect.real
        rows_norm = np.linalg.norm(eig_vect, axis=1, ord=2)
        Y = (eig_vect.T / rows_norm).T
        self.labels_ = self.__cluster_method(Y, self.n_clusters)

        return self
