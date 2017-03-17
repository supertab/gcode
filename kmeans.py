import numpy as np
from numpy import random as rd
import gen_random_dot as dot
import scipy.cluster.vq as vq

# dot.dot_plot( np.array(D), col='ro')

def my_kmeans(D, k=3):
    # 初始集合中心
    # m_k = dot.init_centroid(k) 
    m_k = dot.init_c(D, k)
    dot.dot_plot( np.array(D), dstyle='o')
    dot.dot_plot(m_k, 'rx')

    # m_k = (np.abs(50*np.random.randn(3,2))).tolist() 
    vector_len=len(D[len(D)//2]) # 得到向量的长度
    # 初始化集合
    vector_set = []
    for i in range(k):
        vector_set.append([])

    iter_num = 0
    while iter_num<30:
        iter_num += 1
        # 计算欧式距离, 以欧式距离最min作为聚类原则
        for vec in D:
            vec_ex = np.tile(vec,(len(m_k), 1))
            distance = ((np.array(m_k) - vec_ex)**2).sum(1)
            pos = np.where( distance== distance.min())[0].tolist()
            vector_set[pos[0]].append(vec)

        # 计算每个集合的中心，与m_k中的每个元素对比，存在差异则更新m_k,不存在差异就判断为收敛
        # 求差，如果差小于一定范围则认为两集合的数据点相同
        if_converge = True
        tmp_m =[]
        # for (index, each_vector) in enumerate(vector_set):
        for idx, vec_group in enumerate(vector_set):
            # 计算每个集合的中心
            # tmp_m.append( (np.sum(each_set[0:,0])/len(each_set), np.sum(each_set[0:,1])/len(each_set)) )
            vec_group = np.array(vec_group)
            grp_len = len(vec_group)
            if grp_len == 0:
                tmp_m.append(m_k[idx])
                continue
            new_center = []
            for i in range(vector_len):
                new_center.append(np.sum(vec_group[0:,i])/grp_len)
            tmp_m.append(new_center)
        # 与原始向量集合比较
        # dot.dot_plot( np.array(m_k), col='o')
        mse = np.sqrt( ((np.array(tmp_m)-m_k)**2).sum())
        if mse > 1e-5:
            m_k = tmp_m
            if_converge= False

        if if_converge: 
            break
    print(iter_num)
    return m_k 

D= dot.data(n=100)
k = 3
m_kz = my_kmeans(D, k)
m_kscipy, idx = vq.kmeans(D, k)



dot.dot_plot( np.array(m_kz))
dot.dot_plot(np.array(m_kscipy), dstyle='^')
