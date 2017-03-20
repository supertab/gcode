import numpy as np
import scipy.cluster.vq as vq

def lbg_vq(data, k, split_factor=1e-3): 
    centorid = data.sum(axis=0)/ data.shape[0]
    for i in range(0,k):
        centorid = np.r_['0,2', centorid-split_factor, centorid+split_factor]
        centorid, label = vq.kmeans2(data, centorid, minit='matrix')
    return centorid

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    ndot = 1000 
    # data = np.random.randn(ndot*2).reshape((ndot,2))
    data = 100*np.random.rand(ndot*2).reshape((ndot,2))
    split_factor= 0.001
    k = 7
    centorid = lbg_vq(data, k, split_factor)
    # compare with random initial kmeans
    c2, idx = vq.kmeans2(data, 2**k)
    plt.plot(data[:,0], data[:,1], 'bo')
    plt.plot(centorid[:,0], centorid[:,1], 'ro')
    plt.plot(c2[:,0], c2[:,1], 'y^')
    plt.show()
