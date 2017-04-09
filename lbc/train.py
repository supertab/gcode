# get vq dictionary
# read block set from directory
# directory: sample, blocks, vqdict
import os, pickle
import numpy as np
import scipy.cluster.vq as vq
import mahotas as mh

# readall in memory: assume the memory is enough
def _read_all(src, imglist):
    imgs = []
    print('image reading...')
    counter = 0
    for img in imglist:
        imgs.append(mh.imread(src+img, as_grey=True).astype(np.uint8))
        counter +=1
        if counter==100:
            print('read in %d image...'%counter)
            counter=0
    imgs = np.array(imgs)
    return imgs.astype(np.uint8)

def lbg_vq(data, k, split_factor=1e-3): 
    centorid = data.sum(axis=0)/ data.shape[0]
    for i in range(0,k):
        centorid = np.r_['0,2', centorid-split_factor, centorid+split_factor]
        centorid, label = vq.kmeans2(data, centorid, minit='matrix')
    return centorid

def gen_train_set(src, des, imgsize, blksize, readall): 
    image_list = os.listdir(src)
    # make directory
    if des[:-1] not in os.listdir():
        os.mkdir(des)
    # fliter picture
    support_formats = ['.BMP','.JPG']
    image_list = [img for img in image_list if os.path.splitext(img)[1].upper() in support_formats ]
    if readall:
        image_list = _read_all(src, image_list)
    lim = imgsize - blksize + 1
    block_set = []
    # naming of each block_set
    nimg = (imgsize// blksize)**2
    count = '0'*len(str(nimg))
    n_iter = 0
    suffix = '.bks' # set of block which at same position
    for px in range(0, lim, blksize):
        for py in range(0, lim, blksize):
            n_iter +=1
            bkname = count[:-len(str(n_iter))] + str(n_iter)
            for each_img in image_list:
                if not readall:
                    each_img= mh.imread(src+each_img, as_grey=True)
                block_set.append(each_img[px:px+blksize, py:py+blksize])
            print(bkname,'split down ...')
            fname = des+bkname+suffix
            tmp = np.array(block_set)
            # kmeans2 only support float32,64
            tmp = tmp.astype(np.float32)
            with open(fname, 'wb') as f:
                pickle.dump(tmp.reshape((tmp.shape[0], tmp.shape[1]*tmp.shape[2])), f)
            block_set.clear()

def vq_train(src, des, kset, imgsize, blksize=8, gen_set=True, sav=True, allin=False):
    # split image to blocks
    if gen_set:
        gen_train_set(src, des, imgsize, blksize, allin)
    # tree kmeans to generate vq dictionary
    bkset_list= np.array(os.listdir(des))
    bkset_list.sort()
    vqd = []
    for bkset in bkset_list:
        with open(des+bkset, 'rb') as f:
            bk = pickle.load(f)
        vqd.append(lbg_vq(bk, kset))
        print('VQ training %s down ...'%bkset)
    vqd = np.array(vqd)
    vqd = vqd.astype(np.uint8)
    if sav:
        with open('vqdict.pkl', 'wb') as f:
            pickle.dump(vqd, f)
            print('save VQ dictionary down at current director...')
    return vqd


if __name__ == '__main__':
    dirBkset = 'testBlocks/'
    dirSample = r'./'
    imgsize = 512 
    blksize = 8
    kset = 2
    vqdl = vq_train(dirSample, dirBkset, kset, imgsize, blksize, gen_set=True, sav=True)
