

import numpy as np
import pickle
import matplotlib.pyplot as plt

def GenDCT(bb ,Pn=16):
    '''
    params
        bb: block shape
        Pn:
    return
        array, shape:(64, 256)
    '''
    dct = np.zeros((bb,Pn))
    bb = np.arange(bb)
    for k in range(Pn):
        V = np.cos(bb*k*np.pi/Pn)
        if k>0:
            V = V-np.mean(V)
        dct[:,k] = V/np.linalg.norm(V)
        DCT= np.kron(dct,dct)
    return DCT

def GenSin(N, fbeg=1, fend=50, pbeg=-np.pi, pend=np.pi):
    '''
    params
        N: number of sample
        fbeg, gend: the scope of frequency in sin(), Hz
        pbeg, pend: the scope of phase in sin(), Hz
    return
        sindb: uint sin function, array, shape(N, M)
    '''
    x = np.linspace(0,1,N)
    fre_depart = 0.2 #频率分辨率
    fre_count = int((fend-fbeg)// fre_depart)
    pha_depart = 0.2 #相位分辨率
    pha_count =int((pend-pbeg)// pha_depart)
    Sindb=[]
    for fre in np.linspace(fbeg, fend, fre_count, dtype=np.float64):
        for pha in np.linspace(pbeg, pend, pha_count, dtype=np.float64):
            y = np.sin(2*np.pi*fre*x + pha)
            Sindb.append(y/np.linalg.norm(y)) #对库中原子单位化
    return np.array(Sindb).T

def GenGabor(N=64):
    '''
    params
        N: 信号的长度
    return
        array, shape:(64, M)
    '''
    #############################################################
    # 功能:
    # 生成原子长度为N的 gabor 过完备原子库
    # gabor函数:
    # Gr(t)=1/sqrt(s) * G((t-u)/s) * cos(vt+w) -- r=(s, u, v, w)
    # G(t) = e^(-pi*t^2)
    #############################################################
    gabor=[]
    # 参数根据参考文献中数据设置
    a_base=2.0
    j_min=0
    j_max=np.log2(N)
    u_base=1.0/2
    p_min=0
    v_base=np.pi
    k_min=0
    w_base=np.pi/6.0
    i_min=0
    i_max=12
    for j in np.arange(j_min, j_max+1):
        for p in np.arange(p_min, N*2**(-j+1)+1):
            for k in np.arange(k_min, 2**(j+1)+1):
                for i in np.arange(i_min, i_max+1):
                    s=a_base**j
                    u=p*s*u_base
                    v=k*(1.0/s)*v_base
                    w=i*w_base
                    # t=np.array([i for i in range(0,N)])
                    t=np.arange(0,N)
                    t=(t-u)/(s*1.0)
                    g=(1.0/np.sqrt(s))*np.exp(-np.pi*t*t)*np.cos(v*t+w) # gabor function 离散化
                    #print ("raw g: ", g)
                    g=g/np.sqrt(np.sum(g*g)) #对向量单位化
                    gabor.append(g)
    return np.array(gabor).T


def plot_dict(dic, parts=4):
    n = dic.shape[1]
    nl = [int(i) for i in np.linspace(0, n-1, parts) ]
    pics = '2'+str(int(parts/2))
    plt.figure(1)
    for idx, i in enumerate(nl):
        idx+=1
        ppos =pics+str(idx)
        plt.subplot(ppos)
        plt.plot(dic[:,i])
    plt.show()


if __name__=='__main__':
    # dct  = GenDCT(8)
    # gabor = GenGabor(64)
    # sindb = GenSin(1024)
    gabor = GenGabor(256)
    plot_dict(gabor)