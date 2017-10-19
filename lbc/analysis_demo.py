#!/home/zooo/stdpyenv/bin/python
import sys
import pickle
import time
import os, glob
import os.path as pth
sys.path.append('../scripts')
import select_file
import gen_compress
import quota
import cbc
from PIL import Image
import numpy as np


def file_filter(path, types=['bmp', 'jpg', 'png']):
    des_files = []
    for root, dirs, files in os.walk(path, topdown=True):
        for t in types: 
            des_files.extend(glob.glob(root+os.sep+'*.%s'%t))
    return des_files


blksize = 8
k = 5

vqdpath = '.\\vqdict_%d.pkl' % k
train_dir = r'E:\A_IronSample\分类测试\冷轧带钢\init\train_set'
# test_dir = r'E:\A_IronSample\固融带钢\固融压入异物'
test_dir = r'E:\A_IronSample\分类测试\冷轧带钢\init\测试集\划伤'
kset = cbc.gen_kset(k, train_dir, blksize, 'bmp')
# get imgpath
imgpaths = file_filter(test_dir)

# train
cbc.train(train_dir, kset, blksize, cluster='LBG')

# encode and decode
for imgpath in imgpaths:
    imgdir, imgname = pth.split(imgpath)
    cbcpath = os.path.join(imgdir, 'output', pth.splitext(imgname)[0]+'.cbc')
    imgcbc, calres = cbc.encode(imgpath, vqdpath, blksize, mp=1, mode='rmse')
    cbc.decode(cbcpath, vqdpath, blksize)
    print(imgpath, 'done...')



# k = 5
# vqdpath = '.\\vqdict_%d.pkl' % k
# imgpath = 'E:\\A_IronSample\\西南铝_印痕_1\\normal\\EastAlum\\train\\output\\1002.bmp'
# cbcpath = 'E:\\A_IronSample\\西南铝_印痕_1\\normal\\EastAlum\\train\\output\\1002.cbc'
# cbcsize = cbc.calcucbc(vqdpath, cbcpath)[0]
# print(k)
# bmpsize = 1024 * 1024
# print('compress rate: %.3f' % (bmpsize / cbcsize))
# j2k, j2ksize = gen_compress.bmp2j2k(imgpath, cbcsize)
# jpg, jpgsize = gen_compress.bmp2jpg(imgpath, cbcsize)
# cbcimg = np.array(Image.open(cbcpath.replace('.', '_') + '.bmp'))
# bmp = np.array(Image.open(imgpath))
# print('ssim j2k:', quota.mssim(bmp, np.array(j2k)), j2ksize)
# print('ssim cbc:', quota.mssim(bmp, cbcimg), cbcsize)


'''
compare with jpg2000: compress rate; image quality; compress time;
'''
'''
psnr = lambda im0, im1: 10*np.log10( 255/np.sqrt(((im1-im0)**2).mean()))

blksize = 8
k = 3
srcDIR = 'G:\\图像库\\北海固溶二号线图像\\二号线成卷图像\\003652bmp'
vqdpath = '.\\vqdict_%d.pkl'%k
imgpath = 'G:\\图像库\\北海固溶二号线图像\\二号线成卷图像\\tmp\\22.bmp'
cbcpath = 'G:\\图像库\\北海固溶二号线图像\\二号线成卷图像\\tmp\\output\\22.cbc'

cbcimg, cbcsize = cbc.encode(imgpath, vqdpath, blksize, mode='rmse')
time.sleep(1)
cbc.decode(cbcpath, vqdpath, blksize)
# initial image size
bmpsize = cbcimg[3][0]*cbcimg[3][1]
print('compress rate: %.3f'%(bmpsize/cbcsize))

j2k, j2ksize = gen_compress.bmp2j2k(imgpath, cbcsize)

print(cbcsize, j2ksize)
'''

'''
try:
    img_name = testDIR+select_file.select_file(img_format, destdir=testDIR)
except:
    print('illega choose...')


with open('vqdict.pkl','rb') as f:
    vqds = pickle.load(f)
kset = [k]*len(vqds)

# compress image
cbc.encode.encode(img_name, vqds, kset, imgsize, blksize, show=False)
# convert hexcode to bitstream
# bitstream  = cbc.decode.hex2bit(hexcode)
bitstream = bits.read(img_name[:-4] + '.cbc')
# compress rate
cbcsize = len(bitstream)//(8*1024)
compress_rate = float('%.1f'%(imgsize/ cbcsize))
# decode
img_cbc = cbc.decode.decode(bitstream, vqds, kset, imgsize, blksize)
# sav image
cbc.decode.img_show_save(img_cbc, img_name, sav=True)

img0= Image.open(img_name)
img_bmp = np.array(img0)
img_cbc = img_cbc.astype(np.uint8)
psnr_cbc =  psnr(img_bmp, img_cbc)

img_jpg, jpgsize = gen_compress.bmp2jpg(img_name, cbcsize)
img_jpg = np.array(img_jpg)
psnr_jpg = psnr(img_bmp, img_jpg)

img_j2k, j2ksize = gen_compress.bmp2j2k(img_name, cbcsize)
img_j2k = np.array(img_j2k)
psnr_j2k = psnr(img_bmp, img_j2k)

print('psnr: cbc %.3f, jpg %.3f, j2k %.3f'%(psnr_cbc, psnr_jpg, psnr_j2k))
'''
'''
#plot
plt.gray()
fig = plt.figure(figsize=(13,10))
plt.text(0.35, 1.1,'IMG: %s, Compress_rate: %s'%(img_name, compress_rate))
plt.axis('off')
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
pic_cbc = fig.add_subplot('224')
plt.axis('off')
plt.title('cbc PSNR:%.2f'%psnr_cbc)
plt.imshow(img_cbc)
fig.subplots_adjust(wspace=0.1, hspace=0.3)
imgf = os.path.splitext(img_name)[0].replace('/', '_')+'.png'
fig.savefig(imgf)
print('save test result: %s'%imgf)
'''
