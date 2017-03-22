import numpy as np
import pickle
import scipy.cluster.vq as vq
import mahotas as mh

def _im2col(img, blk_size):
    cols = []
    lim = img.shape[0] - blk_size +1
    for px in range(0, lim, blk_size):
        for py in range(0, lim, blk_size):
            col = img[px:px+blk_size, py:py+blk_size]
            cols.append( col.reshape(blk_size*blk_size))
    return np.array(cols)

# int2str
# convert int to bit string then connect them
def _int2bitstr(idxset, kset):
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

def _bit2hex(bitstream, fout):
    # 4 bit a group, save as hex
    len_bs = len(bitstream)
    hexcode = ''
    for i in range(0, len_bs, 4):
        hextmp = hex(int(bitstream[i:i+4], 2))
        hexcode += hextmp[2]
    # save to txt
    with open(fout[:-3]+'hex', 'w') as f:
        f.write(hexcode)
    return hexcode

def encode(img_name, vqds, kset, imgsize, blksize=8):
    # read image
    img = mh.imread(img_name)
    if img.shape[0] != imgsize:
        print('illegal image size ...')
        return None
    if len(img.shape) == 3:
        img = mh.colors.rgb2gray(img)
    # split
    cols = _im2col(img, blksize)
    cols = cols.astype(np.float64)
    # vector quantization
    label = []
    err = []
    for col, vqd in zip(cols, vqds):
        lb, dist = vq.vq(np.array([col]), vqd)
        label.append(lb[0])
        err.append(dist[0])
    label = np.array(label)
    # label list to bit stream
    bitstream = _int2bitstr(label, kset)
    hexcode = _bit2hex(bitstream, img_name)
    print('%s is encodeing to: %s\nHEXCODE: %s'%(img_name,\
            img_name[:-3]+'hex', hexcode))
    return hexcode

if __name__ == '__main__':
    img_name = '0220.bmp'
    vqd_name = 'vqdict.pkl'
    # loadf vq dictionary set
    with open(vqd_name, 'rb') as f:
        vqds = pickle.load(f)
    kset = [2]*len(vqds)
    blksize = 8
    imgsize = 512
    hexcode = encode(img_name, vqds, kset, imgsize, blksize)
