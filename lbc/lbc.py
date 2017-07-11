import sys; sys.path.append('../scripts')
import mp_dicts, mp2, quota
import numpy as np
import os, pickle, time
import os.path as pth
import scipy.cluster.vq as vq
from PIL import Image
import matplotlib.pyplot as plt


def _int2bitstr(idxset, kset):
    # int2str
    # convert int to bit string then connect them
    # idxset: the index of each vector in VQ dictionary
    # kset: k bits to represent the VQ dictionary
    bs = ''
    for i,c in zip(idxset, kset):
        tmp = bin(i)[2:] # remove 0b
        bitlen = len(tmp)
        if bitlen < c:
            nzero = c - bitlen
            tmp = '0'*nzero + tmp
        bs += tmp
    return bs

def _bit2hex(bitstream):
    # 4 bit a group, save as hex
    len_bs = len(bitstream)
    hexcode = ''
    for i in range(0, len_bs, 4):
        hextmp = hex(int(bitstream[i:i+4], 2))
        hexcode += hextmp[2]
    return hexcode

def _hex2bit(fin):
    # parse hex stream
    if '.' in fin:
        with open(fin) as f:
            hexstr = f.read()
    else: hexstr=fin
    bitstream = ''
    for each_hex in hexstr:
        # hex to binary
        bits = bin(int(each_hex, 16))[2:]
        # put in bitjar
        bits = '0'*(4-len(bits)) + bits
        bitstream += bits
    return bitstream

def _bitstr2int(bs, kset):
    # bs: bit stream generate by the VQ index
    # kset: k bits to represent the VQ dictionary
    idx = []
    i = 0
    for c in kset:
        tmp = bs[i: i+c]
        idx.append(int(tmp,2))
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
    support_formats = ['.BMP','.JPG','.PNG','.J2K','.JPEG']
    imglist = [img for img in os.listdir(imgdir) if pth.splitext(img)[1].upper() in support_formats ]
    for img in imglist:
        # imgs.append(mh.imread(src+img, as_grey=True).astype(np.uint8))
        imgs.append(np.array(Image.open(imgdir+img).convert('L')).astype(np.uint8))
        counter +=1
        if counter%100==0:
            print('read in %d image...'%counter)
    imgs = np.asarray(imgs)
    print('%d images read in...'%counter)
    return imgs.astype(np.uint8)

def img_split(image_list, pos, blksize): 
    '''
    image_list: 图片矩阵
    pos: 小块的起始位置, (px, py) 二维矩阵
    blksize: 块尺寸, 1个数字,表示正方形块的长和宽
    '''
    py, px = pos
    blkset =[]
    for each_img in image_list:
        # 如果有 n 张图片, 得到 nx8x8 的矩阵
        blkset.append(each_img[py:py+blksize, px:px+blksize])
    # kmeans2 only support float32,64
    blkset = np.array(blkset).astype(np.float32)
    blkset = blkset.reshape(len(image_list), blksize*blksize)
    return blkset

def lbg_vq(data, k, split_factor=1e-3): 
    centorid = data.sum(axis=0)/ data.shape[0]
    for i in range(0,k):
        centorid = np.r_['0,2', centorid-split_factor, centorid+split_factor]
        centorid, label = vq.kmeans2(data, centorid, minit='matrix')
    return centorid

def im2col(img, blksize):
    cols = []
    lim_h = img.shape[0] - blksize +1
    lim_w = img.shape[1] - blksize +1
    for py in range(0, lim_h, blksize):
        for px in range(0, lim_w, blksize):
            col = img[py:py+blksize, px:px+blksize]
            cols.append( col.reshape(blksize*blksize))
    return np.asarray(cols).astype(np.float64)

def col2im(cols, imgsize, blksize):
    img =np.zeros(imgsize)
    lim_h = imgsize[0] - blksize+1
    lim_w = imgsize[1] - blksize+1
    ncol = 0
    for py in range(0, lim_h, blksize):
        for px in range(0, lim_w, blksize):
            img[py:py+blksize, px:px+blksize]=cols[ncol].reshape(blksize, blksize)
            ncol+=1
            #if ncol==3: return np.array(img)
    # 直接转化为 uint8 当数大于 255会从头开始计算, 因此先手动将大于255的数设置为255
    imgf = np.array(img)
    imgf[imgf>255]=255
    return imgf.astype(np.uint8)

def matchv(vqds, cols, mp, mode):
    # set match function threhold
    if mode=='ssim':
        matchfunc = quota.ssim
        threhold = 0.95
    elif mode =='pha':
        matchfunc=quota.pha
        threhold = 0
    elif mode == 'rmse':
        matchfunc=quota.rmse
        threhold = 5
    DCT = mp_dicts.GenDCT(8)
    # match
    vq_idxs, mp_sets, insert_pos =[], [], []

    for col_pos, vcpair in enumerate(zip(vqds, cols)):
        vqd, col = vcpair
        vq_idx, dist = matchfunc(*vcpair)
        if mp==True:
            if dist > threhold:
                mp_vec = mp2.encode(DCT, col).flatten()
                nonz_pos = mp_vec.nonzero()[0] # 得到非零元素的位置
                nonz_val = mp_vec[nonz_pos]    # 得到非零元素的值
                mp_set = list(zip(nonz_pos, nonz_val)) # mp_set: [(idx1, coeff1), (idx2, coeff2), ...] 
                insert_pos.append(col_pos)
                mp_sets.append(mp_set)
                print('f', dist)
            else:
                vq_idxs.append(vq_idx)
        else:
            vq_idxs.append(vq_idx)
    return vq_idxs, mp_sets, insert_pos 

def train(srcDIR, k, blksize, vqs_name='vqdict.pkl'):
    '''
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
    cnt =0
    for py in range(0, lim_h, blksize):
        for px in range(0, lim_w, blksize):
            cnt+=1
            blkset = img_split(imglist, (py,px), blksize)
            vq = lbg_vq(blkset, k)
            vqs.append(vq)
            print('training block %d down...'%cnt)
    vqs = np.asarray(vqs).astype(np.uint8)
    # 保存结果
    with open(vqs_name, 'wb') as f:
        pickle.dump(vqs, f)
    print('save VQ dictionary down...')

def encode(imgpath, vqdpath, blksize, kset, mp=True, mode='rmse'):
    '''
     编码过程:
     量化图片
        |--分块
        |--得到索引
     保存索引
        |--十进制转二进制
        |--连接二进制
        |--保存成文件
    '''
    with open(vqdpath, 'rb') as f:
        vqds = pickle.load(f)
    kset = [kset]*vqds.shape[0]
    img = np.array(Image.open(imgpath))
    cols = im2col(img, blksize)
    vq_idxs, mp_sets, insert_pos = matchv(vqds, cols, mp, mode)
    # bitstream = _int2bitstr(idxs, kset) 
    # hexcode = _bit2hex(bitstream)
    pack = [vq_idxs, mp_sets, insert_pos, img.shape] # 存储图片的尺寸信息
    # make dir if result not exist
    imgdir, imgname = pth.split(imgpath)
    resdir = imgdir+pth.sep+'output'+pth.sep
    if 'output' not in os.listdir(imgdir): os.mkdir(resdir)
    lbcname = resdir + pth.splitext(imgname)[0]+'.lbc'
    with open(lbcname,'wb') as f:
        pickle.dump(pack, f)
    return pack

def recon_vec(D, mp_sets):
    '''
    params:
        D: 列为向量长度，行为个数 64x256
        mp_sets: [[(idx1, coeff1), (idx2, coeff2), ...], [...] ]
    return:
        vectors Nx64
    '''
    vectors=[]
    for each_set in mp_sets:
        vec = np.zeros(64)
        for pos, coef in each_set:
            vec += D[:, pos]*coef
        vectors.append(vec)
    return vectors

def decode(lbcpath, vqdpath, blksize, kset):
    with open(lbcpath, 'rb') as f:
        vq_idxs, mp_sets, insert_pos, imgsize = pickle.load(f)
    with open(vqdpath, 'rb') as f:
        vqds = pickle.load(f)
        vqds = np.delete(vqds, insert_pos, axis=0) # 删除无匹配的vq字典
    kset = [kset]*vqds.shape[0]
    # bitstream = _hex2bit(hexcode)
    # idxs = _bitstr2int(bitstream, kset)
    cols =[]
    # 在还原的时候没有按照对应元素还原
    for vqd, idx in zip(vqds, vq_idxs):
        cols.append(vqd[idx])
    # insert reconstruct vectors back
    dct = mp_dicts.GenDCT(8)
    vecs = recon_vec(dct, mp_sets)
    for v_pos, vec in zip(insert_pos, vecs):
        cols.insert(v_pos, vec)
    img = col2im(cols, imgsize, blksize)
    img2 = Image.fromarray(img)
    imgname = pth.splitext(lbcpath)[0]+'_lbc.bmp'
    img2.save(imgname)
    return img 

def main():
    blksize=8
    k=3
    srcDIR = 'E:\\WZ\\gcode\\testIMG\\stdIMG\\256'
    # srcDIR = 'G:\\图像库\\北海固溶二号线图像\\二号线成卷图像\\003652bmp'
    vqdpath = '.\\vqdict_%d.pkl'%k
    imgpath = '..\\testIMG\\stdIMG\\256\\lenna.bmp'
    # imgpath = 'G:\\图像库\\北海固溶二号线图像\\二号线成卷图像\\tmp\\flaw01.bmp'
    lbcpath = '..\\testIMG\\stdIMG\\256\\output\\lenna.lbc'
    # lbcpath = 'G:\\图像库\\北海固溶二号线图像\\二号线成卷图像\\tmp\\output\\flaw01.lbc'
    encode(imgpath,vqdpath, blksize, k, mp=True)
    time.sleep(2)
    decode(lbcpath, vqdpath, blksize, k)

if __name__ == '__main__':
    main()
   