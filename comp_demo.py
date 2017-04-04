import sys, pickle, time, os
sys.path.append('./lbc')
import lbc, gen_compress
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

psnr = lambda im0, im1: 10*np.log10( 255/np.sqrt(((im1-im0)**2).mean()))

blksize = 8
imgsize = 512
k = 5
img_name = 'testIMG/0006.bmp'
sampleDIR = 'sample/'
bksetDIR = 'blocks/'

with open('vqdict.pkl','rb') as f:
    vqds = pickle.load(f)
kset = [k]*len(vqds)

# compress image
hexcode = lbc.encode.encode(img_name, vqds, kset, imgsize, blksize, show=False)
# convert hexcode to bitstream
bitstream  = lbc.decode.hex2bit(hexcode)
img_lbc = lbc.decode.decode(bitstream, vqds, kset, imgsize, blksize)
# sav image
lbc.decode.img_show_save(img_lbc, img_name, sav=True)

img0= Image.open(img_name)
img_bmp = np.array(img0)
img_lbc = img_lbc.astype(np.uint8)
psnr_lbc =  psnr(img_bmp, img_lbc)

img_jpg, jpgsize = gen_compress.bmp2jpg(img_name, 2.5)
img_jpg = np.array(img_jpg)
psnr_jpg = psnr(img_bmp, img_jpg)

img_j2k, j2ksize = gen_compress.bmp2j2k(img_name, 2.5)
img_j2k = np.array(img_j2k)
psnr_j2k = psnr(img_bmp, img_j2k)

#plot
plt.gray()
fig = plt.figure(figsize=(10,10))
pic_bmp = fig.add_subplot('221')
plt.axis('off')
plt.title('initial')
plt.imshow(img_bmp)
pic_jpg = fig.add_subplot('222')
plt.axis('off')
plt.title('jpg PSNR:%.2f'%psnr_jpg)
plt.imshow(img_jpg)
pic_j2k = fig.add_subplot('223')
plt.axis('off')
plt.title('j2k PSNR:%.2f'%psnr_j2k)
plt.imshow(img_j2k)
pic_lbc = fig.add_subplot('224')
plt.axis('off')
plt.title('lbc PSNR:%.2f'%psnr_lbc)
plt.imshow(img_lbc)
fig.subplots_adjust(wspace=0.2, hspace=0.2)
fig.savefig('test1.png')
