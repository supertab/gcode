import numpy as np
from PIL import Image as im

def psnr(im0, im1):
    mse = np.sqrt(((im1-im0)**2).mean())
    print( 10*np.log10(255/ mse))
    return 10*np.log10(255/ mse)
