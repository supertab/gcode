import numpy as np
import numpy.linalg as linalg
import mahotas as mh
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from MPerr import *

# 使用im2col的一个原因：
# 经过im2col后矩阵中每列的长度由窗口决定，而不是输入图像本身决定
# 因此字典中每个原子的长度只要与窗口的长度符合就行了
def im2col(A, BSZ, stepsize=1):
    # Parameters
    # A 转入矩阵，BSZ: 窗的大小
    # 窗口按行滑动，每个小窗内的数据，按行拼接后作为新矩阵的列
    # bug: stepsize不等于1的时候结果不符合预期
    m,n = A.shape
    s0, s1 = A.strides    
    nrows = m-BSZ[0]+1
    ncols = n-BSZ[1]+1
    shp = BSZ[0],BSZ[1],nrows,ncols
    strd = s0,s1,s0,s1
    out_view = np.lib.stride_tricks.as_strided(A, shape=shp, strides=strd)
    return out_view.reshape(BSZ[0]*BSZ[1],-1)[:,::stepsize]


def col2im(im_size,blocks, bb):
    # 重构图像
    # genrate pos of every pixes by Image and blocks
    x_lim = im_size[0] - bb
    y_lim = im_size[1] - bb
    # padding
    recon_image = np.zeros(im_size)
    weight = np.zeros(im_size)
    col= 0
    for x_pos in range(x_lim+1):
        # for y_pos,each_block in zip(range(y_lim+1), blocks): # lead Error
        for y_pos in range(y_lim+1):
            # padding block
            recon_image[x_pos:x_pos+bb, y_pos:y_pos+bb] += blocks[:,col].reshape((bb,bb))
            # record weight
            weight[x_pos:x_pos+bb, y_pos:y_pos+bb] += np.ones((bb,bb))
            col += 1
    return recon_image / weight


'''
#test split and col2im
sizeA = (4,8)
A = np.arange(32).reshape((4,8))
bb = (2,2)
blocks = im2col(A, bb)
B = col2im(sizeA, blocks, bb[0])
'''

def GenDCT(bb ,Pn=16):
    dct = np.zeros((bb,Pn))
    bb = np.arange(bb)
    for k in range(Pn):
        V = np.cos(bb*k*np.pi/Pn)
        if k>0:
            V = V-np.mean(V)
        dct[:,k] = V/linalg.norm(V)
        DCT= np.kron(dct,dct)
    return DCT

from MPerr import *
from MP import * 
def SparseCodeing(im, block_size, codebook):
    blocks = im2col(im, [block_size,block_size])
    threshold = 1000
    iter_n = 3
    # Coef = MP(codebook, blocks, iter_n)
    Coef = MPerr(codebook, blocks, threshold)
    return Coef

def Decode(coef, im_size, block_size, codebook):
    blocks = np.dot(codebook, coef)
    im = col2im(im_size, blocks, block_size)
    return im

if __name__ == '__main__':
    im = mh.imread('lena200.jpg')
    # 转化为灰度图
    if len(im.shape)==3:
        im= mh.colors.rgb2gray(im) # float64
    # use PIL to convert gray
    # im = im.convert(mode='L') # int8
    bb = 8 # 8x8方形窗口[bb,bb]
    codebook = GenDCT(bb)
    print('Image Encoding ...')
    coeff = SparseCodeing(im, bb, codebook)
    sparse_coef = coo_matrix(coeff) # compress sparse matrix by scipy.sparse's function
    print('Image reconstruct ...')
    recovery_im = Decode(coeff, im.shape, bb, codebook)

    # plot
    plt.gray()
    plt.subplot(121)
    plt.title('initial image')
    plt.imshow(im)
    plt.subplot(122)
    plt.title('reconstruct image')
    plt.imshow(recovery_im)
plt.show()
