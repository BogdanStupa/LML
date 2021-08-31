import numpy as np

from  typing import Callable



def compute_affinity(
    X: np.array,
    kernel: Callable[[np.array, np.array, float], np.array],
    sigma: float
):
    n = X.shape[0]
    A = np.ones(shape=(n, n))
    for i in range(n):
        for j in range(i + 1, n):
            A[i][j] = A[j][i] = kernel(X[i], X[j], sigma)
    return A


