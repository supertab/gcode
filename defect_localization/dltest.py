# dl: defect localization
#!/home/zooo/stdpyenv/bin/python3
import numpy as np
from PIL import Image
import scipy.cluster.vq as vq
import pickle

def map_a2b(num, a=[-10.22, 20.31], b=[0, 255]):
    min_a, max_a = min(a), max(a)
    min_b, max_b = b[0], b[1]
    range_a = max_a - min_a
    range_b = max_b - min_b
    # mapping num from a to b
    tmp =  ((num-min_a)/range_a) * range_b + min_b
    # round
    return int('%.0f'%tmp)

def im2col(img, blk_size):
    cols =[]
    lim = img.shape[0] - blk_size +1
    for px in range(0, lim, blk_size):
        for py in range(0, lim, blk_size):
            col = img[px:px+blk_size, py:py+blk_size]
            cols.append(col.reshape(blk_size*blk_size))
    return np.array(cols)

# 计算 PSNR
def psnr(arr1, arr2):
    mse = ((arr1-arr2)**2).mean()
    return 10*np.log10(255/np.sqrt(mse))

with open('vqdict.pkl', 'rb') as f:
    vqds = pickle.load(f)
im = Image.open('1901.bmp')
im = np.array(im)
cols = im2col(im, 8)
idx, dist = [], []
psnrs =[]
for col, vqd in zip(cols, vqds):
    i, d = vq.vq(np.array([col]), vqd)
    idx.append(i[0])
    dist.append(d[0])
    psnrs.append(psnr(col, vqd))

min_p, max_p = min(psnrs), max(psnrs)
lower, upper = np.floor(min_p), np.ceil(max_p)
cnt =[]
for i in range(int(lower), int(upper)+1):
    cnt.append(0)

for i in psnrs:
    cnt[map_a2b(i, [min_p, max_p], [lower, upper]) -int(lower)] +=1

def cnt_date(dataIn):
    cnt = []
    min_d, max_d = int(min(dataIn)), int(max(dataIn))
    for i in range(min_d, max_d+1):
        cnt.append(0)
    for i in dataIn:
        cnt[int(i)-min_d] +=1
    return cnt

# 检测 vq 的 dist 为多少, 理论上为最小欧式距离
col0=cols[0]
vq16=vqds[0][16]
rse = np.sqrt(np.sum((col0-vq16)**2))
