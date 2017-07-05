import numpy as np
from PIL import Image
import sys, os, glob

# 从命令行中读入一张图片， 分离三个通道的像素值

img_name = sys.argv[-1]
im = np.array(Image.open(img_name))
if im.ndim>1:
    for ch in range(im.ndim):
        im_ch = im[:,:,ch]
        tmp_img = Image.fromarray(im_ch)
        tmp_img.save(os.path.splitext(img_name)[0]+'_'+str(ch)+os.path.splitext(img_name)[1])
print('split down...')

