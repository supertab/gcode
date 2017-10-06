import numpy as np
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt

def find_eps(vec, k):
    '''
    vec: 列向量形式的样本, 每列为一个样本向量
    k: MinPnt
    功能: 通过图像确定eps的值, 根据经验
    '''

    kdist=[]
    for coln in range(vec.shape[1]):
        tmp = (vec - vec[:,coln].reshape((vec.shape[0],1)))**2
        tmp = np.sort(np.sqrt(tmp.sum(0)))
        kdist.append(tmp[k])
    y = np.sort(kdist)[::-1]
    plt.grid(axis='y')
    plt.plot(y)
    plt.show()


if __name__ == '__main__':
    centers = [[-1, 1], [1, 1], [-1, -1]]
    X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
                                random_state=0)
    find_eps(X.T, 12)
