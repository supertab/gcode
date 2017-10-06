import sys
import os
sys.path.append('../scripts')
import pickle
import gen_compress
import quota
import lbc
from PIL import Image
import numpy as np
# 比较cbc， cbc-mp, jpg, jpg2000的压缩效果
# 以 cbc-mp算法的压缩比为参照

imgname = 'test/0027.bmp'
imgpath = 'test/%s' % os.path.split(imgname)[1]
vqdpath = 'test/r_vqdict_5.pkl'
blksize = 8


cbcname = imgname.replace(os.path.splitext(imgname)[1], '.cbc')
cbc_mp_name = imgname.replace(os.path.splitext(imgname)[1], '.cbcmp')
# read vqd, bmp
img_cbc, calres0 = lbc.encode(imgpath, vqdpath, blksize, mp=0, mode='rmse')
img_cbc_mp, calres = lbc.encode(imgpath, vqdpath, blksize, mp=1, mode='rmse')
with open(cbcname, 'wb') as f:
    pickle.dump(img_cbc, f)

with open(cbc_mp_name, 'wb') as f:
    pickle.dump(img_cbc_mp, f)

cbc_size = calres0[0]
cbc_mp_size = calres[0]

bmp0 = Image.open(imgpath)
bmp = np.array(bmp0)
cbc = lbc.decode(cbcname, vqdpath, blksize)
cbc_mp = lbc.decode(cbc_mp_name, vqdpath, blksize)
j2k, j2ksize = gen_compress.bmp2j2k(imgpath, cbc_mp_size)
jpg, jpgsize = gen_compress.bmp2jpg(imgpath, cbc_mp_size)

ssim_cbc = quota.mssim(bmp, cbc)
ssim_cbc_mp = quota.mssim(bmp, cbc_mp)
ssim_jpg = quota.mssim(bmp, np.array(jpg))
ssim_j2k = quota.mssim(bmp, np.array(j2k))

bmpsize = bmp0.size[0] * bmp0.size[1]
print('cbc: %.3f  %.3f' % (bmpsize / cbc_size, ssim_cbc))
print('cbc_mp: %.3f  %.3f' % (bmpsize / cbc_mp_size, ssim_cbc_mp))
print('jpg: %.3f  %.3f' % (bmpsize / jpgsize, ssim_jpg))
print('j2k: %.3f  %.3f' % (bmpsize / j2ksize, ssim_j2k))
