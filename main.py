import numpy as np
import copy

from mylearn.cluster import KMeans
from mylearn.utilities.plot_clustered_2d_sample import plot_clustered_2d_sample


def generate_random_sample_of_normal_destribution(mean, cov, N):
    return np.stack((np.random.multivariate_normal(mean, cov, N).T), axis=-1)


def generate_base_sample_with_k_clusters(K, dim, N=20):
    total_number_of_points = N * K.shape[0]
    base_sample = np.ndarray(shape=(total_number_of_points, dim))
    base_labels = np.ndarray(shape=(total_number_of_points,))
    i, index = 0, np.random.choice(total_number_of_points, total_number_of_points, replace=False)
    for k, [mean, cov] in enumerate(K):
        coordinates = generate_random_sample_of_normal_destribution(mean, cov[0], N)
        for coordinate in coordinates:
            base_sample[index[i]] = coordinate
            base_labels[index[i]] = k + 1
            i += 1
    return base_sample, base_labels


def generate_random_number_of_points_from_base_sample(base_sample, number_of_points):
    idx = np.random.choice(base_sample.shape[0], number_of_points, replace=False)
    sample = copy.deepcopy(base_sample[idx])
    return sample


distribution = np.array([
    [[-2, 5], [np.array([[1, 0], [0, 10]])]],
    [[5, -5], [np.array([[10, 0], [0, 1]])]],
    [[3, 3], [np.array([[20, 0], [0, 12]])]]
])

K = distribution.shape[0]
N = 1000
dimension = 2
base_sample, base_labels = generate_base_sample_with_k_clusters(distribution, dimension, N)


# [[5, -5, 3], [[10, 0, 0], [0, 1, 0], [0, 0, 7]]],
#     [[5, -5, 3], [[10, 0, 0], [0, 1, 0], [0, 0, 7]]],
#     [[5, -5, 3], [[10, 0, 0], [0, 1, 0], [0, 0, 7]]],
#     [[5, -5, 3], [[10, 0, 0], [0, 1, 0], [0, 0, 7]]],


sample = generate_random_number_of_points_from_base_sample(base_sample, int(N * .15))


kmeans = KMeans().fit(sample)



print(kmeans.predict(np.array([[22, 33], [1, 2], [-2, -5], [0, 0]])))

plot_clustered_2d_sample(sample, kmeans.get_labels(), kmeans.get_centers())


# print(kmeans)