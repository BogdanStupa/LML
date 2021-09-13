import matplotlib.pyplot as plt
import numpy as np

from mylearn.cluster.spectral import MySpectralClustering
from mylearn.cluster.kmeans import MyKMeans

from mylearn.cluster.tests \
    import test_clustering_with_normal_distribution_data, test_clustering_with_circle_data



test_clustering_with_circle_data(cluster=MyKMeans)
test_clustering_with_normal_distribution_data(cluster=MyKMeans)

# test_clustering_with_circle_data(cluster=MySpectralClustering)
# test_clustering_with_normal_distribution_data(cluster=MySpectralClustering)


