import numpy as np
from scipy.signal import convolve2d
from PIL import Image

'''
psnr, mssim
'''
psnr = lambda im0, im1: 20*np.log10( 255/np.sqrt( ((im0-im1)**2).mean()))
veclen = lambda vec: np.sqrt((vec**2).sum()) #calculate the length of vector

def window_gaussian(size, sigma=0.5):
    n = (size-1)//2
    var=np.asarray([ [x**2 + y**2 for x in range(-n, n+1)] for y in range(-n,n+1) ])
    hg = np.exp(-var/(2*sigma**2))
    h = hg/hg.sum()
    return h

def mssim(img1, img2):
    '''
    params
        img0, img1: nxn array, float
    return
        index of ssim
    '''
    img1=img1.astype(np.float64)
    img2=img2.astype(np.float64)
    L = 255
    K = [0.01, 0.03]
    C1 = (K[0]*L)**2
    C2 = (K[1]*L)**2
    window = window_gaussian(11, 1.5)
    mu1 = convolve2d(img1, window, 'valid')
    mu2 = convolve2d(img2, window, 'valid')
    mu1_sq, mu2_sq = mu1**2, mu2**2
    mu1_mu2 = mu1*mu2
    sigma1_sq = convolve2d(img1*img1, window, 'valid') - mu1_sq
    sigma2_sq = convolve2d(img2*img2, window, 'valid') - mu2_sq
    sigma12 = convolve2d(img1*img2, window, 'valid') - mu1_mu2
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    m_ssim = ssim_map.mean()
    # print(mssim)
    return m_ssim

if __name__ == '__main__':
    img1 = np.array(Image.open('lenna.bmp'))
    img2 = np.array(Image.open('lenna2.bmp'))
    print(mssim(img1, img2))


