import os

import numpy as np
import imageio as imageio
import matplotlib.pyplot as plt


def make_gif_2d_clusterd_sample(X, n_clusters, data_for_gif):
    fnames = []
    i = 0
    for l, c in data_for_gif:
        i += 1
        fname = f"k{i}.png"
        fig, ax = plt.subplots(figsize=(10, 10))
        fig.suptitle(f"Iteration {i}")

        SX = [[] for j in range(n_clusters)]
        SY = [[] for j in range(n_clusters)]
        CX, CY = np.ndarray(shape=(n_clusters,)), np.ndarray(shape=(n_clusters,))

        for j, point in enumerate(X):
            SX[l[j] - 1].append(point[0])
            SY[l[j] - 1].append(point[1])

        for j in range(n_clusters):
            CX[j] = c[j][0]
            CY[j] = c[j][1]

        for j in range(n_clusters):
            ax.scatter(SX[j], SY[j], s=50)

        for j in range(n_clusters):
            ax.scatter(CX[j], CY[j], s=130, c='r', marker='*')

        fig.savefig(fname)
        plt.close()
        fnames.append(fname)

    with imageio.get_writer('media/kmeans.gif', mode='I') as writer:
        for filename in fnames:
            image = imageio.imread(filename)
            for j in range(20):
                writer.append_data(image)

    for filename in set(fnames):
        os.remove(filename)