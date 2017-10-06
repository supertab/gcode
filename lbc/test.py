import sys
sys.path.append('../scripts')
import mp_dicts
import todb
import mp2
import quota
import numpy as np
import os
import pickle
import time
import glob
import sys
import os.path as pth
import scipy.cluster.vq as vq
from PIL import Image
import matplotlib.pyplot as plt

im = Image.open('..\\testIMG\\stdIMG\\256\\lenna.bmp')
ima = np.array(im)
ima = ima.T
imb = Image.fromarray(ima.T)
imc = imb.rotate(-90)
imc.save('test.bmp')
