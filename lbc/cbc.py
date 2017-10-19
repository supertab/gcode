import sys
sys.path.append('../scripts')
import mp_dicts
import mp2
import sqlite3
# import todb
import quota
import numpy as np
import os
import pickle
import time
import glob
import sys
import os.path as pth
import scipy.cluster.vq as vq
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN


def _int2bitstr(idxset, kset):
    # int2str
    # convert int to bit string then connect them
    # idxset: the index of each vector in VQ dictionary
    # kset: k bits to represent the VQ dictionary
    bs = ''
    for i, c in zip(idxset, kset):
        tmp = bin(i)[2:]  # remove 0b
        bitlen = len(tmp)
        if bitlen < c:
            nzero = c - bitlen
            tmp = '0' * nzero + tmp
        bs += tmp
    return bs


def _bit2hex(bitstream):
    # 4 bit a group, save as hex
    len_bs = len(bitstream)
    hexcode = ''
    for i in range(0, len_bs, 4):
        hextmp = hex(int(bitstream[i:i + 4], 2))
        hexcode += hextmp[2]
    return hexcode


def _hex2bit(fin):
    # parse hex stream
    if '.' in fin:
        with open(fin) as f:
            hexstr = f.read()
    else:
        hexstr = fin
    bitstream = ''
    for each_hex in hexstr:
        # hex to binary
        bits = bin(int(each_hex, 16))[2:]
        # put in bitjar
        bits = '0' * (4 - len(bits)) + bits
        bitstream += bits
    return bitstream


def _bitstr2int(bs, kset):
    # bs: bit stream generate by the VQ index
    # kset: k bits to represent the VQ dictionary
    idx = []
    i = 0
    for c in kset:
        tmp = bs[i: i + c]
        idx.append(int(tmp, 2))
        i += c
    return np.array(idx)


def read_all(imgdir):
    # readall in memory: assume the memory is enough
    '''
    输入: 图片所在目录的路径
    输出: 目录中所有图片组成的array, 图片均转换为8位灰度图
    '''
    if imgdir[-1] != pth.sep:
        imgdir += pth.sep
    imgs = []
    print('image reading...')
    counter = 0
    support_formats = ['.BMP', '.JPG', '.PNG', '.J2K', '.JPEG']
    imglist = [img for img in os.listdir(imgdir) if pth.splitext(img)[
        1].upper() in support_formats]
    for img in imglist:
        # imgs.append(mh.imread(src+img, as_grey=True).astype(np.uint8))
        imgs.append(
            np.array(Image.open(imgdir + img).convert('L')).astype(np.uint8))
        counter += 1
        if counter % 100 == 0:
            print('read in %d image...' % counter)
    imgs = np.asarray(imgs)
    print('%d images read in...' % counter)
    return imgs.astype(np.uint8)


def img_split(image_list, pos, blksize):
    '''
    功能: 对一组图片进行分块, 将相同位置的块保存在同一个列表中
    image_list: 图片矩阵
    pos: 小块的起始位置, (px, py) 二维矩阵
    blksize: 块尺寸, 1个数字,表示正方形块的长和宽
    '''
    py, px = pos
    blkset = []
    for each_img in image_list:
        # 如果有 n 张图片, 得到 nx8x8 的矩阵
        blkset.append(each_img[py:py + blksize, px:px + blksize])
    # kmeans2 only support float32,64
    blkset = np.array(blkset).astype(np.float32)
    blkset = blkset.reshape(len(image_list), blksize * blksize)
    return blkset

# Cluster
def lbg_vq(data, k, split_factor=1e-3):
    centorid = data.sum(axis=0) / data.shape[0]
    for i in range(0, k):
        centorid = np.r_['0,2', centorid -
                         split_factor, centorid + split_factor]
        centorid, label = vq.kmeans2(data, centorid, minit='matrix')
    return centorid


def my_dbscan(X, eps, min_pnt):
    '''
    params:
        X: 输入样本, np.array, 样本按行排(N,P), 行数为样本的个数N, 列数为样本的维度P
        eps: 密度半径, 使用find_eps.py, 绘图选取
        min_pnt: 经验公式: >=维度+1
    return:
        centers: 重心: (均值, 最大长度向量和最小长度向量)(按行排)
        clusters: dbscan的聚类结果-每个类的样本情况(按行排)
    '''

    db = DBSCAN(eps=eps, min_samples=min_pnt).fit(X)
    labels = db.labels_
    ncluster = len(set(labels)) - (1 if -1 in labels else 0)
    centers, clusters = [], []
    for k in range(ncluster):
        # vec是按行排的
        tmp_set = []
        idx = np.where(labels==k)[0]
        vecs = X[idx]
        # 取均值, 最大长度向量和最小长度向量作为重心
        tmp_set.append(vecs.mean(axis=0))
        vecs_len = np.linalg.norm(vecs, axis=1)
        tmp_set.append(vecs[np.argmax(vecs_len)])
        tmp_set.append(vecs[np.argmin(vecs_len)])
        centers.extend(tmp_set)

        clusters.append(vecs)
    return np.array(centers), np.array(clusters)

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
    '''
    imgsize: 为np.array的shape值, 和Image对象中的图片尺寸是反的
    '''
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


def matchv(vqds, DCT, cols, mp, mode):
    # set match function threhold
    if mode == 'ssim':
        matchfunc = quota.ssim
        threhold = 0.95
    elif mode == 'pha':
        matchfunc = quota.pha
        threhold = 0
    elif mode == 'rmse':
        matchfunc = quota.rmse
        threhold = 10
    # match
    vq_idxs, mp_sets, insert_pos = [], [], []
    for col_pos, vcpair in enumerate(zip(vqds, cols)):
        vqd, col = vcpair
        vq_idx, dist = matchfunc(*vcpair)
        if mp == 1:
            if dist > threhold:
                mp_vec = mp2.encode(DCT, col).flatten()
                nonz_pos = mp_vec.nonzero()[0]  # 得到非零元素的位置
                nonz_val = mp_vec[nonz_pos]    # 得到非零元素的值
                # mp_set: [(idx1, coeff1), (idx2, coeff2), ...]
                mp_set = list(zip(nonz_pos, nonz_val))
                insert_pos.append(col_pos)
                mp_sets.append(mp_set)
                # print('f', dist)
            else:
                vq_idxs.append(vq_idx)
        else:
            vq_idxs.append(vq_idx)
    return vq_idxs, mp_sets, insert_pos


def train(srcDIR, kset, blksize, pkg_name='vqdict.pkl', cluster='LBG'):
    '''
    cluster: 聚类方法: LBG 和 DBSCAN
    # 训练过程:
    # 分块:
    #  |--读取图像: 图像可以时长方形, 但高和宽必须为8的倍数
    #  |  |--方式1(_read_all): 将图片全部加载进内存,再处理,速度快,但需要占用大内存.
    #  |  |--方式2(): 每做一次分块, 读取一次图像, 占用小内存, 频繁读取图片,因此速度慢.
    #  |--分块(img_split): 从每张图的相同位置分块, 分块顺序和解码顺序要相同, 正方形无重叠分块
    # 量化: 每分好一个小块后就进行聚类,得到量化表
    #  |--
    # 存储:
    '''
    imglist = read_all(srcDIR)
    imgsize = imglist.shape[1:]
    # 按顺序分块聚类
    lim_h = imgsize[0] - blksize + 1
    lim_w = imgsize[1] - blksize + 1
    vqs = []
    cnt = 0
    nkset = [] # 使用dbscan聚类时用到
    for py in range(0, lim_h, blksize):
        for px in range(0, lim_w, blksize):
            blkset = img_split(imglist, (py, px), blksize)
            if cluster=='LBG':
                # use LBG cluster 
                vq = lbg_vq(blkset, kset[cnt])
            elif cluster=='DBSCAN':
                # use DBSCAN cluster
                vq, clus = my_dbscan(blkset, 45, 68)
                # 计算k
                k_tmp = np.log2(len(vq)) 
                k2 = int(k_tmp) + (1 if int(k_tmp)!=k_tmp or int(k_tmp)==0 else 0)
                nkset.append(k2)
            vqs.append(vq) # 样本按行排
            cnt += 1
            print('training block %d down...' % cnt)
    vqs = np.asarray(vqs).astype(np.uint8)
    # 更新kset
    if len(nkset):
        kset = nkset
    # 将 kset，vqs，dct打包
    pkg = [kset, vqs, mp_dicts.GenDCT(8)]
    # 保存结果
    with open(pkg_name, 'wb') as f:
        pickle.dump(pkg, f)
    print('save VQ dictionary down...')


def encode(imgpath, vqdpath, blksize, mp=True, mode='rmse'):
    '''
    Parameters
    ----------
    imgpath: string, 目标图片路径
    vqdpath: string, 解析包路径
    blksize: 分块大小
    mp: bool, 是否使用MP重构图块
    mode: string, 相似度判断方法

    Returns
    ------
    img_lbc: list
    [vq_idxs, mp_sets, insert_pos, img.shape]
    out2: float
    the size of img_lbc, byte
    '''
    with open(vqdpath, 'rb') as f:
        vqds_pkg = pickle.load(f)
    kset, vqds, dct = vqds_pkg
    img = np.array(Image.open(imgpath))
    cols = im2col(img, blksize)
    vq_idxs, mp_sets, insert_pos = matchv(vqds, dct, cols, mp, mode)
    # bitstream = _int2bitstr(idxs, kset)
    # hexcode = _bit2hex(bitstream)
    img_lbc = [vq_idxs, mp_sets, insert_pos, img.shape]  # 存储图片的尺寸信息
    # make dir if result not exist
    imgdir, imgname = pth.split(imgpath)
    resdir = imgdir + pth.sep + 'output' + pth.sep
    if 'output' not in os.listdir(imgdir):
        os.mkdir(resdir)
    lbcname = resdir + pth.splitext(imgname)[0] + '.cbc'
    with open(lbcname, 'wb') as f:
        pickle.dump(img_lbc, f)
    return img_lbc, calcuLBC(vqds_pkg, img_lbc)


def recon_vec(D, mp_sets):
    '''
    params:
        D: 列为向量长度，行为个数 64x256
        mp_sets: [[(idx1, coeff1), (idx2, coeff2), ...], [...] ]
    return:
        vectors Nx64
    '''
    vectors = []
    for each_set in mp_sets:
        vec = np.zeros(64)
        for pos, coef in each_set:
            vec += D[:, pos] * coef
        vectors.append(vec)
    return vectors


def decode(lbcpath, vqdpath, blksize):
    with open(lbcpath, 'rb') as f:
        vq_idxs, mp_sets, insert_pos, imgsize = pickle.load(f)
    with open(vqdpath, 'rb') as f:
        vqds_pkg = pickle.load(f)
        kset, vqds, dct = vqds_pkg
        kset = np.delete(kset, insert_pos)
        vqds = np.delete(vqds, insert_pos, axis=0)  # 删除无匹配的vq字典
    # bitstream = _hex2bit(hexcode)
    # idxs = _bitstr2int(bitstream, kset)
    cols = []
    # 在还原的时候没有按照对应元素还原
    for vqd, idx in zip(vqds, vq_idxs):
        cols.append(vqd[idx])
    # insert reconstruct vectors back
    # dct = mp_dicts.GenDCT(8)
    vecs = recon_vec(dct, mp_sets)
    for v_pos, vec in zip(insert_pos, vecs):
        cols.insert(v_pos, vec)
    img = col2im(cols, imgsize, blksize)
    img2 = Image.fromarray(img)
    imgname = pth.splitext(lbcpath)[0] + '_cbc.bmp'
    img2.save(imgname)
    return img


def calcuLBC(codebook, lbc_img):
    '''
    params
        codebook(np.array): kset(W*H维向量)+vqds(所有vq小块的字典集合)+dct(256x64离散余弦字典)
                知道vqds通过计算每个vqd的长度就能得到kset
        lbc_img(np.array):  [vq_idxs, mp_sets, insert_pos, img.shape]
    return
        字节数
    '''
    # 判断类型，如果传入文件名就打开文件
    if type(codebook) != type(lbc_img):
        print('type err...')
        return
    if isinstance(codebook, str):
        with open(lbc_img, 'rb') as f:
            lbc_img = pickle.load(f)
        with open(codebook, 'rb') as f:
            codebook = pickle.load(f)
    vq_idxs, mp_sets, insert_pos = lbc_img[:3]
    kset, vqds, dct = codebook
    kset, vqds = np.delete(kset, insert_pos), np.delete(
        vqds, insert_pos, axis=0)
    vqIdx, atomInfo, insertPos, atomNum = 0, 0, 0, 0
    # 计算vq_idx比特串长度r
    for k in kset:
        vqIdx += k
    # 计算insertPos,atomNum,atomInfo
    # vqIdx=0
    N0 = len(kset)
    N1 = len(insert_pos)
    insertPos = 18 * N1
    atomNum = 4 * N1
    atomInfo = 24 * sum([len(i) for i in mp_sets])
    size = vqIdx + insertPos + atomInfo + atomNum
    # print('N0: %s, N1: %s' % (N0, N1))
    return np.ceil(size / 8), N0, N1


def gen_kset(k, srcDir, blksize, fmat='bmp'):
    imglist = srcDir + os.sep + '*.%s'%fmat
    if len(glob.glob(imglist)):
        imgpath = glob.glob(imglist)[0]
    else:
        return
    img = np.array(Image.open(imgpath).convert('L'))
    nblk = (img.shape[0] / blksize) * (img.shape[1] / blksize)
    kset = [k] * int(nblk)
    return kset


def main():
    usage = '''Usage: %run lbc.py v1 v2 func
    ------------------------
    v1: set K value, scope: 2-10
    v2: if use MP, scope: 0,1
    func: train, encode, decode
    '''
    if len(sys.argv) < 3:
        print(usage)
        return
    blksize = 8
    k = int(sys.argv[1])
    # srcDIR = '..\\testIMG\\stdIMG\\512'
    srcDIR = 'E:\\A_IronSample\\固融带钢\\trainset'
    vqdpath = '.\\vqdict_%d.pkl' % k
    # imgpath = '..\\testIMG\\stdIMG\\512\\lenna.bmp'
    # imgpath = 'C:\\Users\\supertab\\Desktop\\金属样本\\hot_iron_test\\0129.jpg'
    # imgpath = 'E:\\A_IronSample\\固融带钢\\trainset\\000001.bmp'
    imgpath = 'E:\\A_IronSample\\固融带钢\\固融划伤\\iron_001.bmp'
    imgdir, imgname = pth.split(imgpath)
    # lbcpath = '..\\testIMG\\stdIMG\\512\\output\\lenna.lbc'
    # lbcpath = 'C:\\Users\\supertab\\Desktop\\金属样本\\hot_iron_test\\output\\0129.lbc'
    lbcpath = imgdir + '\\output\\' + pth.splitext(imgname)[0] +'.cbc'
    kset = gen_kset(k, srcDIR, blksize, 'bmp')
    func = 'train' if len(sys.argv) < 4 else sys.argv[3]
    if func == 'train':
        train(srcDIR, kset, blksize, cluster='DBSCAN')
        # time.sleep(3)
        os.rename("vqdict.pkl", 'vqdict_%d.pkl' % k)
    elif func == 'encode':
        stime = time.time()
        img_lbc, calres = encode(
            imgpath, vqdpath, blksize, mp=int(sys.argv[2]), mode='rmse')
        etime = time.time()
        time_consume = etime - stime  # 以秒为单位
        print('time consum:', time_consume)
        lbcsize, N0, N1 = calres
        print('N0: %s, N1: %s' % (N0, N1))
        print('cbcsize:', lbcsize)
        time.sleep(2)
        decode(lbcpath, vqdpath, blksize)
        # 计算ssim
        img0 = np.array(Image.open(imgpath).convert('L'))
        img1 = np.array(Image.open(lbcpath.replace(
            '.cbc', '_cbc.bmp')).convert('L'))
        mssim = quota.mssim(img0, img1)
        print('ssim:', mssim)
        # name, k, ssim, filesize, time, n0, n1 插入数据库
        # con = sqlite3.connect('test_result.db')
        # cur = con.cursor()
        # tbname = 'cbc_mp' if sys.argv[2] == '1' else 'cbc'
        # insert
        # insert_sql = '''insert into %s (name, k, ssim, filesize, n0, n1)
        # values (?, ?, ?, ?, ?, ?);''' % tbname
        # update
        # update_sql = '''update %s set time=? where name=\'lena512\' and k=?''' % tbname
        # try:
            # cur.execute(insert_sql, ('lena512', k, mssim, lbcsize, N0, N1))
            # cur.execute(update_sql, (time_consume, k))
            # con.commit()
        # except:
            # con.close()
    else:
        print('err...')
        return


if __name__ == '__main__':
    main()
