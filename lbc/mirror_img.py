import os
import numpy as np
from PIL import Image
import glob

imgpath = 'E:\\A_IronSample\\铝板1024x1024\\left\\1.6\\crop'
os.chdir(imgpath)
imglist = glob.glob('*.bmp')
os.mkdir('output')
for i in imglist:
    im = Image.open(i)
    ima = np.array(im)
    im.close()
    imb = Image.fromarray(ima.T)
    imb = imb.rotate(-90)
    imb.save('output'+os.sep+i)
    print(i)
