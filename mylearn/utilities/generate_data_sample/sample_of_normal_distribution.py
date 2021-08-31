import numpy as np
import copy

def generate_random_sample_of_normal_distribution(mean, cov, N):
    return np.stack((np.random.multivariate_normal(mean, cov, N).T), axis=-1)

def generate_base_sample_with_k_clusters(K, dim, N=20):
    total_number_of_points = N * K.shape[0]
    base_sample = np.ndarray(shape=(total_number_of_points, dim))
    base_labels = np.ndarray(shape=(total_number_of_points,))
    i, index = 0, np.random.choice(total_number_of_points, total_number_of_points, replace=False)
    for k, [mean, cov] in enumerate(K):
        coordinates = generate_random_sample_of_normal_distribution(mean, cov[0], N)
        for coordinate in coordinates:
            base_sample[index[i]] = coordinate
            base_labels[index[i]] = k + 1
            i += 1
    return base_sample, base_labels


def generate_random_number_of_points_from_base_sample(base_sample, number_of_points):
    idx = np.random.choice(base_sample.shape[0], number_of_points, replace=False)
    sample = copy.deepcopy(base_sample[idx])
    return sample