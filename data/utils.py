import numpy as np

def batchify(X, batch_size, y=None):
    l = len(X)
    for ndx in range(0, l, batch_size):
        if y is None:
            yield X[ndx:min(ndx + batch_size, l)]
        yield X[ndx:min(ndx + batch_size, l)], y[ndx:min(ndx + batch_size, l)]

def normalize_unit_sphere(points):
    centroid = np.mean(points, axis=0)
    points -= centroid
    furthest_distance = np.max(np.sqrt(np.sum(abs(points) ** 2, axis=-1)))
    points /= furthest_distance
    return points