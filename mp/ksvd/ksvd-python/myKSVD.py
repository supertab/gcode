import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.linear_model import orthogonal_mp_gram

def ompcode2(D, X, maxIter=16, maxErr=1000, pflag=False):
    '''
    encoding with mp_err.
    params
        D: dict
    return
        A:
    '''  
    # n: the length of very vector
    # P: the number of vector in X
    # K: the number of vector in D
    x_shape = X.shape
    if len(x_shape) > 1:
        n, P = x_shape
    else:
        n, P = x_shape[0], 1
        X = X.reshape(n, 1)
    K = D.shape[1]
    # 系数矩阵，行数等于D中向量的个数，列数等于X中向量的个数
    A = np.zeros((K, P))
    D_T = D.T
    
    count = []
    for k in range(P):
        indx = []
        cofficient = []
        x = X[:,k]
        residual = x
        for j in range(maxIter): # max iterate time is 100
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
            if (residual**2).sum() < maxErr:
                break
        count.append(j)
        # 将系数存入系数矩阵A中
        for i,j in zip(indx,cofficient):
            A[i][k] = j # k为信号矩阵的第k个向量
    if pflag:
        print( np.histogram(count))
    return A

def ompcode(D, X, T):
    gram = np.dot(D.T, D);
    cov = np.dot(D.T, X.T);
    return orthogonal_mp_gram(gram, cov, T, None);

def ksvd(Y, K, T):
    # Y: 训练数据 （P * N)
    # K: 字典中元素的数量 
    # T: 非零元素的最大个数
    global D, X;
    
    maxIter = 50;
    maxErr = 0.1;
    
    (P, N) = Y.shape;
    D = np.mat(np.random.rand(P, K));
    Yt = Y.T;
    # 归一化，使得 Di 的 L2-norm 为 1
    for i in range(K): 
        D[:,i] /= np.linalg.norm(D[:,i])
    J = 0;
    while ( J < maxIter):
        # 使用 OMP 计算出当前 D 对应的稀疏表达        
        X = ompcode(D,Yt,T);
        for i in range(0, K):
            # 用到的样本的列表
            usedXi = np.nonzero(X[i,:])[0];
            # 如果 Di 没有被任何样本用到
            if (len(usedXi) == 0):
                    continue;     
            # 去掉 DiXiT 
            tmpX = X;
            tmpX[i,:] = 0;
            # 计算 ER
            ER = Y[:,usedXi] - np.dot(D,tmpX[:,usedXi])
            U, s, V = np.linalg.svd(ER)
            # 更新 X 和 D
            X[i,usedXi] = s[0]*V[0,:]
            D[:,i] = U[:,0]

        
        E = Y - np.dot(D,X)
        Enorm = [ np.linalg.norm(E[:,i]) for i in range(0, N)]
        err = np.max(Enorm)
        print('Iter: %d, err: %.3f' % (J, err));
        if (err < maxErr):
            break;
        
        J += 1;
        
    return D;


def im2col(img, blksize):
    cols = []
    lim_h = img.shape[0] - blksize + 1
    lim_w = img.shape[1] - blksize + 1
    for py in range(0, lim_h, blksize):
        for px in range(0, lim_w, blksize):
            col = img[py:py + blksize, px:px + blksize]
            cols.append(col.reshape(blksize * blksize))
    return np.asarray(cols).astype(np.float64)


def col2im(cols, imgsize, blksize):
    img = np.zeros(imgsize)
    lim_h = imgsize[0] - blksize + 1
    lim_w = imgsize[1] - blksize + 1
    ncol = 0
    for py in range(0, lim_h, blksize):
        for px in range(0, lim_w, blksize):
            img[py:py + blksize, px:px +
                blksize] = cols[ncol].reshape(blksize, blksize)
            ncol += 1
            # if ncol==3: return np.array(img)
    # 直接转化为 uint8 当数大于 255会从头开始计算, 因此先手动将大于255的数设置为255
    imgf = np.array(img)
    imgf[imgf > 255] = 255
    return imgf.astype(np.uint8)

if __name__ == '__main__':
    img = np.array(Image.open('lena.bmp'))
    imgshape = img.shape
    img0 = im2col(img, 8)
    D = ksvd(img0.T, 100, 2)
    # reconstruct
    m = img0.mean(axis=0)
    Yt = img0-m
    X = ompcode(D, Yt, 3)
    Y = np.dot(D, X).T
    Y += m
    img1 = col2im(Y, imgshape, 8)
    # show img
    f, (ax0, ax1) = plt.subplots(1,2)
    plt.gray()
    ax0.imshow(img)
    ax0.set_title('img0')
    ax1.imshow(img1)
    ax1.set_title('img1')
    plt.show()

