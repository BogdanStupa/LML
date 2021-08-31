import numpy as np

from mylearn.cluster.base_cluster import BaseCluster

from mylearn.utilities.generate_data_sample import generate_base_sample_with_k_clusters
from mylearn.utilities.generate_data_sample.generate_circles import generate_n_circles
from mylearn.utilities.plot_data_sample.plot_clustered_2d_sample import plot_clustered_2d_sample


# test with circles data
def test_clustering_with_circle_data(
        cluster,
        radiuses=np.array([1, 4, 7, 10]),
        noises=np.array([1.5, 1, 1.5, .3]),
        center=np.array([2, 4])
) -> None:
    if not issubclass(cluster, BaseCluster):
        raise TypeError("cluster must be subclass of BaseCluster")
    X_data_circles = generate_n_circles(radiuses=radiuses, noises=noises, number_of_points=150) + center
    plot_clustered_2d_sample(X_data_circles)

    spectral_clustering = cluster(n_clusters=radiuses.shape[0]).fit(X_data_circles)
    plot_clustered_2d_sample(X_data_circles, spectral_clustering.labels_)




#test with data with normal distribution
def test_clustering_with_normal_distribution_data(
        cluster,
        number_of_points_in_one_cluster=300,
        distribution=np.array([
            [[-3, 5], [np.array([[1, 0], [0, 10]])]],
            [[6, -7], [np.array([[10, 0], [0, 1]])]],
            [[8, 8], [np.array([[2, 0], [0, 12]])]]
        ]),
        dimension=2
):
    if not issubclass(cluster, BaseCluster):
        raise TypeError("cluster must be subclass of BaseCluster")
    base_sample, base_labels = generate_base_sample_with_k_clusters(distribution, dimension, number_of_points_in_one_cluster)
    plot_clustered_2d_sample(base_sample)

    spectral_clustering = cluster(n_clusters=distribution.shape[0]).fit(base_sample)
    plot_clustered_2d_sample(base_sample, spectral_clustering.labels_)



# sample = generate_random_number_of_points_from_base_sample(base_sample, int(N * .15))
# kmeans = KMeans(make_gif=True).fit(sample)
#
# print(kmeans.predict(np.array([[22, 33], [1, 2], [-2, -5], [0, 0]])))
# plot_clustered_2d_sample(sample, kmeans.get_labels(), kmeans.get_centers())