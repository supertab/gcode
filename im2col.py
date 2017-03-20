import os, pickle
import mahotas as mh
import numpy as np

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


if __name__ == '__main__':
    dirBkset= 'blocks/' # directroy to store splited blocks
    dirSample='sample/'
    image_type = '.bmp'
    image_size = 128
    block_size = 8
    im2col(dirSample, dirBkset)
