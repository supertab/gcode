import sys
usage='''Usage: python3 ssim_std_cmd.py img1.jpg img2.jpg '''
if len(sys.argv) != 3:
    print(usage)
    exit()
import numpy as np 
from scipy.signal import convolve2d
from PIL import Image

def window_gaussian(size, sigma=0.5):
    n = (size-1)//2
    var=np.asarray([ [x**2 + y**2 for x in range(-n, n+1)] for y in range(-n,n+1) ])
    hg = np.exp(-var/(2*sigma**2))
    h = hg/hg.sum()
    return h

def readImg2grey(imgName):
    im = Image.open(imgName)
    if im.mode !='L':
        im = np.array(im.convert('L')).astype(np.float32)
    else:
        im = np.array(im).astype(np.float32)
    return im

def mssim(imgName1, imgName2, cvmode='valid'):
    L = 255
    K = [0.01, 0.03]
    C1 = (K[0]*L)**2
    C2 = (K[1]*L)**2
    window = window_gaussian(11, 1.5)
    # window = window_gaussian(3, 1)
    img1 = readImg2grey(imgName1)
    img2 = readImg2grey(imgName2)
    mu1 = convolve2d(img1, window, cvmode)
    mu2 = convolve2d(img2, window, cvmode)
    mu1_sq, mu2_sq = mu1**2, mu2**2
    mu1_mu2 = mu1*mu2
    sigma1_sq = convolve2d(img1*img1, window, cvmode) - mu1_sq
    sigma2_sq = convolve2d(img2*img2, window, cvmode) - mu2_sq
    sigma12 = convolve2d(img1*img2, window, cvmode) - mu1_mu2
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    m_ssim = ssim_map.mean()
    # print(mssim)
    return m_ssim

def main():


    img1 = sys.argv[1]
    img2 = sys.argv[2]
    print('MSSIM: %.4f'%(mssim(img1, img2)))

if __name__ == '__main__':
    main()

