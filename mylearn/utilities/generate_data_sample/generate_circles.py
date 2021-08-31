import numpy as np


def generate_circle(radius: int, number_of_points: int = 100, center: np.array = np.array([0, 0]),
                    noise=1) -> np.ndarray:
    theta = np.linspace(0, 2 * np.pi, number_of_points)
    x = radius * np.cos(theta) + np.random.uniform(0, noise, number_of_points) + center[0]
    y = radius * np.sin(theta) + np.random.uniform(0, noise, number_of_points) + center[1]

    return np.stack((x, y), axis=-1)


def generate_n_circles(
        radiuses: np.array,
        noises: np.array,
        number_of_points: int = 200,
        center: np.array = np.array([0, 0])
) -> np.array:
    res = np.array([center])
    n = radiuses.shape[0]
    for i in range(n):
        sample = generate_circle(radius=radiuses[i], number_of_points=number_of_points, center=center, noise=noises[i])
        res = np.concatenate((res, sample), axis=0)
    return res
