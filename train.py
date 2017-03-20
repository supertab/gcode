# get vq dictionary
# read block set from directory
# directory: sample, blocks, vqdict
import os, pickle
import numpy as np
import scipy.cluster.vq as vq
import mahotas as mh

def lbg_vq(data, k, split_factor=1e-3): 
    centorid = data.sum(axis=0)/ data.shape[0]
    for i in range(0,k):
        centorid = np.r_['0,2', centorid-split_factor, centorid+split_factor]
        centorid, label = vq.kmeans2(data, centorid, minit='matrix')
    return centorid


def im2col(src, des, imgtype='.bmp', imgsize=128, bksize=8):
    image_list = os.listdir(src)
    # make directory
    if des[:-1] not in os.listdir():
        os.mkdir(des)
    image_list = [img for img in image_list if img.find(imgtype)+1 ]
    lim = imgsize - bksize + 1
    block_set = []
    # naming of each block_set
    nimg = (imgsize// bksize)**2
    count = '0'*len(str(nimg))
    n_iter = 0
    suffix = '.bks' # set of block which at same position
    for px in range(0, lim, bksize):
        for py in range(0, lim, bksize):
            n_iter +=1
            bkname = count[:-len(str(n_iter))] + str(n_iter)
            for each_img in image_list:
                img = mh.imread(src+each_img, as_grey=True)
                block_set.append(img[px:px+bksize, py:py+bksize])
            print(bkname,'...')
            fname = des+bkname+suffix
            tmp = np.array(block_set)
            tmp = tmp.astype(np.float64)
            with open(fname, 'wb') as f:
                pickle.dump(tmp.reshape((tmp.shape[0], tmp.shape[1]*tmp.shape[2])), f)
            block_set.clear()



dirBkset = 'blocks/'
dirVqset = 'vqdict/'
dirSample = 'sample/'
# split image to blocks
# im2col(dirSample, dirBkset)

# tree kmeans to generate vq dictionary
bkset_list= os.listdir(dirBkset)
suffix = '.vqd'
k = 1

if dirVqset[:-1] not in os.listdir():
    os.mkdir(dirVqset)

for bkset in bkset_list:
    with open(dirBkset+bkset, 'rb') as f:
        bk = pickle.load(f)
    vqd = lbg_vq(bk, k)
    with open(dirVqset+bkset[:-4]+suffix, 'wb') as f:
        pickle.dump(vqd, f)



