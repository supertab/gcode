import sys; sys.path.append('../../scripts')
import numpy as np 
import mp_dicts, im_trans, mp2, quota
import matplotlib.pyplot as plt
from PIL import Image
import pickle, os
'''运行一次50分钟, 占用大量内存40G'''

print('generating dictionarys...')
Gabor  = mp_dicts.GenGabor(64)
DCT = mp_dicts.GenDCT(8)
print('prepare image...')
imgpath = '..{psep}..{psep}testIMG{psep}stdIMG{psep}256{psep}lenna.bmp'.format(psep=os.sep)

img0 = np.array(Image.open(imgpath).convert('L'))
imgsize = img0.shape
blksize = (8,8)
imgcols = im_trans.im2col(img0, blksize)

class imgdata:
    '''test...
    pickle dump 类的时候只能保存实例的数据
    '''

    # recon_img=[]
    # psnr=[]
    # ssim=[]
    def __init__(self, info, img0):
        self.tips=info
        self.init_img=img0
        self.recon_img=[]
        self.psnr=[]
        self.ssim=[]
    def read(self, img1):
        self.recon_img.append(img1)
        self.psnr.append(quota.psnr(self.init_img, img1))
        self.ssim.append(quota.mssim(self.init_img, img1))

dct_data = imgdata('dct', img0)
gabor_data = imgdata('gabor', img0)

for n in range(10, 60, 5):
    print('decomposing image with DCT...', n)
    coefs1 = mp2.encode(DCT, imgcols, maxErr=10, maxIter=n)
    print('decomposing image with Gabor...', n)
    # coefs2 = mp2.encode(Gabor, imgcols, maxErr=10, maxIter=n)
    print('construct image...', n)
    img1 = mp2.decode(coefs1, imgsize, 8, DCT)
    # img2 = mp2.decode(coefs2, imgsize, 8, Gabor)
    dct_data.read(img1)
    gabor_data.read(img2)

fname = 'output%sdata_'%os.sep+str(n)+'.pkl'
with open(fname, 'wb') as f:
    pickle.dump((dct_data, gabor_data), f)


# 画图使用同目录下的plot.py文件

# plt.gray()
# plt.imshow(imgdct)
# plt.show()
