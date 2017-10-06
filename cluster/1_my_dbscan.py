import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt


def my_dbscan(X, eps, min_pnt):
    db = DBSCAN(eps=eps, min_samples=min_pnt).fit(X)
    labels = db.labels_
    ncluster = len(set(labels)) - (1 if -1 in labels else 0)
    centers, clusters = [], []
    for k in range(ncluster):
        idx = np.where(labels==k)[0]
        vecs = X[idx]
        clusters.append(vecs)
        centers.append(vecs.mean(axis=0))
    return np.array(centers), np.array(clusters)

if __name__ == '__main__':
    centers = [[1, -1], [-1, 1], [1, 1], [-1, -1]]
    X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
                                random_state=0)
    centers, clusters = my_dbscan(X, 0.21, 12)
    plt.plot(X[:,0], X[:,1], 'o')
    plt.plot(centers[:,0], centers[:,1], 'ro')
    plt.show()
