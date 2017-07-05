#!/bin/python
import numpy as np
from PIL import Image
import sys

# 说明: 计算两幅图的 PSNR
# 过程: 程序只接收命令行参数, 将程序拷贝到图片所在目录
# 从命令行中传入两幅图像的名称
# 输出: PSNR

# open image
im0 = np.array(Image.open(sys.argv[1]))
im1 = np.array(Image.open(sys.argv[2]))

psnr = lambda im0, im1: 20*np.log10( 255/np.sqrt( ((im0-im1)**2).mean()))

print('PSNR:', psnr(im0, im1))
