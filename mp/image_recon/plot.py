'''
read imgdata class from .pkl file
1. show image
2. plot 
'''
import numpy as np 
import matplotlib.pyplot as plt 
import pickle,os
import matplotlib as mpl
mpl.rc('font', family='SimHei')

class imgdata:
    pass

fname = 'output%sdata_55.pkl'%os.sep
with open(fname, 'rb') as f:
    img_dct, img_gabor = pickle.load(f)

# print(img_dct.ssim)
# print(img_gabor.ssim)

# show img
def showimg(fname='test.png'):
    '''matplotlib库的保存图片，多个子图
    子图调整：去坐标，设置画布大小，标题
    '''
    f, (ax0, ax1, ax2) = plt.subplots(1,3) # ,igsize=(5,2.8)只能单方向（单参数）调整
    # plt.figure(figsize=(5,4))
    # f.set_figure(figsize=(5,4))
    f.subplots_adjust(wspace=0.1) #可以为负值 
    plt.gray()
    ax0.imshow(img_dct.init_img)
    ax0.axis('off')
    ax0.text(106, 280, '(a)', fontsize=18)
    # ax1.set_title('dct')
    ax1.imshow(img_dct.recon_img[2])
    ax1.axis('off')
    ax1.text(112, 280, '(b)', fontsize=18)
    # ax2.set_title('gabor')
    ax2.imshow(img_gabor.recon_img[2])
    ax2.axis('off')
    ax2.text(112, 280, '(c)', fontsize=18)
    # f.savefig(fname)
    plt.show()

def plot_line():
    '''
    '''
    x = [i for i in range(10, 60, 5)]
    y1 = img_dct.psnr
    y2 = img_gabor.psnr
    f2 = plt.figure(2)
    dct_line, = plt.plot(x,y1, '-o', label='DCT')
    gabor_line, = plt.plot(x,y2, '-^', label='Gabor')
    plt.legend(handles=[dct_line, gabor_line])
    plt.xlabel('原子个数'), plt.ylabel('PSNR')
    plt.show()

# showimg()
plot_line()
