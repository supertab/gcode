import numpy as np
from scipy.signal import convolve2d
from scipy.cluster import vq
from PIL import Image

'''
psnr, mssim
'''
psnr = lambda im0, im1: 20 * np.log10(255 / np.sqrt(((im0 - im1)**2).mean()))
veclen = lambda vec: np.sqrt((vec**2).sum())  # calculate the length of vector


def window_gaussian(size, sigma=0.5):
    n = (size - 1) // 2
    var = np.asarray([[x**2 + y**2 for x in range(-n, n + 1)]
                      for y in range(-n, n + 1)])
    hg = np.exp(-var / (2 * sigma**2))
    h = hg / hg.sum()
    return h


def mssim(img1, img2, gaussw=11):
    '''
    params
        img0, img1: nxn array, float
    return
        index of ssim
    '''
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    L = 255
    K = [0.01, 0.03]
    C1 = (K[0] * L)**2
    C2 = (K[1] * L)**2
    window = window_gaussian(gaussw, 1)
    mu1 = convolve2d(img1, window, 'valid')
    mu2 = convolve2d(img2, window, 'valid')
    mu1_sq, mu2_sq = mu1**2, mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = convolve2d(img1 * img1, window, 'valid') - mu1_sq
    sigma2_sq = convolve2d(img2 * img2, window, 'valid') - mu2_sq
    sigma12 = convolve2d(img1 * img2, window, 'valid') - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
        ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    m_ssim = ssim_map.mean()
    # print(mssim)
    return m_ssim

# 选取最佳的原子


def ssim(*args):
    '''
    参数：
        vqd：聚类字典
        col：64x1 的向量
    输出：
        best_elem_idx: 与输入的col最匹配的原子的索引
        best_factor: 
    功能：
    接收8x8的小块'''
    vqd, col = args
    block = col.reshape((8, 8))
    best_factor = 0
    for idx, elem in enumerate(vqd):
        factor = mssim(elem.reshape(8, 8), block, gaussw=3)
        if factor > best_factor:
            best_elem_idx = idx
            best_factor = factor
    return best_elem_idx, best_factor


def rmse(*args):
    """vqd,col需要有一方是浮点数"""
    vqd, col = args
    idx, dist = vq.vq(np.array([col]), vqd)
    return idx[0], dist[0] / np.sqrt(col.shape[0])


def main():
    img1 = np.array(Image.open("..\\testIMG\\AlphaIMG\\A3.png").convert('L'))
    img2 = np.array(Image.open("..\\testIMG\\AlphaIMG\\B.png").convert('L'))
    print('MSSIM: %.4f' % (mssim(img1, img2)))

if __name__ == '__main__':
    main()

if __name__ == '__main__':
    img1 = np.array(Image.open('lenna.bmp'))
    img2 = np.array(Image.open('lenna2.bmp'))
    print(mssim(img1, img2))
