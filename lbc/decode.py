import numpy as np
import pickle
import bits

def col2im(cols, img_size, blksize):
    img = np.zeros(img_size)
    limx = img.shape[0] - blksize +1
    limy = img.shape[1] - blksize +1
    ncol = 0
    for px in range(0, limx, blksize):
        for py in range(0, limy, blksize):
            col = cols[ncol].reshape((blksize, blksize))
            img[px:px+blksize, py:py+blksize] = col
            ncol += 1
    return img

# parse hex stream
def hex2bit(fin):
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

def decode(bitstream, vqds, kset, imgshape, blksize=8):
    # generate label of vq dictionary
    label = _bitstr2int(bitstream, kset)
    cols = []
    for vqd, idx in zip(vqds, label):
        cols.append(vqd[idx])
    cols = np.array(cols)
    img = col2im(cols, imgshape, blksize) 
    print('reconstruct image down...')
    return img

def img_show_save(img, fin, show=False, sav=False):
    import matplotlib.pyplot as plt
    import PIL.Image as Image
    # show and save
    if show:
        plt.gray()
        plt.imshow(img)
        plt.show()
    if sav:
        im_out = Image.fromarray(img.astype(np.uint8))
        img_name = fin[:-4]+'_lbc.bmp'
        im_out.save(img_name)
        #print('save image: %s'%img_name)

if __name__== '__main__':
    vqd_name = 'vqdict.pkl'
    # load vq dict
    with open(vqd_name,'rb') as f:
        vqds = pickle.load(f)
    kset = [5]*len(vqds) 
    imgshape = (512, 2048)
    blksize = 8
    fin = '0342_90.bmp'
    # bitstream = hex2bit(fin)
    bitstream=bits.read(fin)
    img = decode(bitstream, vqds, kset, imgshape, blksize)
    img_show_save(img, fin, 0, 1)
