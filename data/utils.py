def batchify(X, batch_size, y=None):
    l = len(X)
    for ndx in range(0, l, batch_size):
        if y is None:
            yield X[ndx:min(ndx + batch_size, l)]
        yield X[ndx:min(ndx + batch_size, l)], y[ndx:min(ndx + batch_size, l)]