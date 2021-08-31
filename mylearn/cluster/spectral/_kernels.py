import numpy as np


def squared_exponential(x: np.array, y: np.array, sigma: float):
    norm = np.linalg.norm(x - y)
    dist = norm * norm
    return np.exp(- dist / (2 * sigma * sigma))
