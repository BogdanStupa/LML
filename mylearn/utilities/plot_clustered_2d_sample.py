import matplotlib.pyplot as plt
import numpy as np


def plot_clustered_2d_sample(
        sample,
        labels=np.array([]),
        centroids=np.array([]),
        fig_title=""
):
    fig, ax = plt.subplots(figsize=(10, 10))

    if not labels.shape[0]:
        labels = np.zeros(shape=sample.shape[0])

    n_clusters = int(np.amax(labels)) or 1

    if n_clusters != centroids.shape[0]:
        raise

    X = [[] for i in range(n_clusters)]
    Y = [[] for i in range(n_clusters)]
    CX, CY = np.ndarray(shape=(n_clusters,)), np.ndarray(shape=(n_clusters,))

    for i, point in enumerate(sample):
        X[labels[i] - 1].append(point[0])
        Y[labels[i] - 1].append(point[1])

    for i in range(centroids.shape[0]):
        CX[i] = centroids[i][0]
        CY[i] = centroids[i][1]

    for i in range(n_clusters):
        ax.scatter(X[i], Y[i], s=50)

    for i in range(n_clusters):
        ax.scatter(CX[i], CY[i], s=130, c='r', marker='^')

    fig.suptitle(fig_title, fontsize=15)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    ax.grid(True)
    plt.show()