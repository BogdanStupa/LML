import numpy as np
# np.seterr(divide='ignore', invalid='ignore')

def _clustering(X, labels, centers):
    print("KMEANS_Clusterin1", X.shape)
    print("cefe", centers)
    for i, point in enumerate(X):
        last_min_distance = np.inf
        for j, centroid in enumerate(centers):
            current_distance = min(last_min_distance, np.linalg.norm(point - centroid))
            if current_distance < last_min_distance:
                last_min_distance = current_distance
                labels[i] = j + 1
    print("KMEANS_Clusterin2")



def _normalize(X):
    max_coordinate = np.amax(X, axis=0)
    min_coordinate = np.amin(X, axis=0)
    diff = max_coordinate - min_coordinate

    if not np.all(diff):
        raise

    for i in range(X.shape[0]):
        X[i] = (X[i] - min_coordinate) / diff

    return min_coordinate, diff

# def _normalize_with_fitted_data(X, min_coordinate, diff):
#     for i in range(X.shape[0]):
#         X[i] = (X[i] - min_coordinate) / diff


def _recalculate_centers(X, labels, centers):
    for i in range(centers.shape[0]):
        centers[i] = np.sum(X[labels == i + 1], axis=0) / labels[labels == i + 1].shape[0]
        print(centers[i], labels[labels == i + 1].shape[0])


def _tolerance(centers, prev_centers, tolerance):
    for i in range(centers.shape[0]):
        if not np.linalg.norm(centers[i] - prev_centers[i]) / np.linalg.norm(centers[i]) < tolerance:
            return False
    return True


def _inertia(
        X,
        sample_weight,
        centers,
        labels
):
    return []