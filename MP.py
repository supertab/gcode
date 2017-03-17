import numpy as np
'''
D - dictionary(normalized)
X - signals to represent
L - the max number of coefficients for each signal
'''

def MP(D,X,L):
# n: the length of very vector
# P: the number of vector in X
# K: the number of vector in D
    x_shape = X.shape
    if len(x_shape) > 1:
        n, P = x_shape
    else:
        n, P = x_shape[0], 1
    K = D.shape[1]
    # 系数矩阵，行数等于D中向量的个数，列数等于X中向量的个数
    A = np.zeros((K, P))
    D_T = D.T

    for k in range(P):
        indx = []
        cofficient = []
        x = X[:,k]
        residual = x
        for j in range(L):
            proj = np.dot(D_T, residual)
            # get max lenth of project and pos
            if abs(proj.min()) >= abs(proj.max()):
                maxL = proj.min()
            else:
                maxL = proj.max()
            pos = np.where(proj==maxL)[0]
            indx.append(pos.tolist()[0]) # array.tolist()转array为数组
            cofficient.append(maxL)
            # bug: D[:,indx[0:j+1]]会出现3个维度，原因是indx里面是array型数据
            x_reconstruct = np.dot(D[:,indx[0:j+1]],np.array( cofficient))
            residual = x - x_reconstruct
            if (residual**2).sum() < 1e-6:
                break
        
        # 将系数存入系数矩阵A中
        for i,j in zip(indx,cofficient):
            A[i][k] = j # k为信号矩阵的第k个向量
    return A
