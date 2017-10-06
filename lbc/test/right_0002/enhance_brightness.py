import numpy as np
from PIL import Image, ImageEnhance
import os
import glob

'''
对当前目录下的图片文件进行转换
'''


def get_imgname():
    img_types = ['.bmp', '.png', '.jpg']
    # find img type in dir
    for i in os.listdir():
        img_type = os.path.splitext(i)[1].lower()
        if img_type in img_types:
            break
    # get all image
    img_list = glob.glob('*' + img_type)
    return img_list


def mkdir(path, srcpath='.'):
    fullpath = srcpath + os.sep + path
    if path in os.listdir(srcpath):
        # try:
        #     os.listdir(fullpath)  # check file or dir
        # except NotADirectoryError:
        #     os.mkdir(fullpath)
        return
    else:
        os.mkdir(fullpath)

# get one channel of image


def get_channel(ch):
    img_list = get_imgname()
    output_dir = str(ch)
    # mkdir(output_dir)
    for img in img_list:
        im = np.array(Image.open(img))
        im_ch = im[:, :, ch]
        Image.fromarray(im_ch).save(
            output_dir + os.sep + os.path.split(img)[1])
        print('split %s down...' % img)


def bright_enhance(factor):
    img_list = get_imgname()
    output_dir = str(factor)
    mkdir(output_dir)
    for imgname in img_list:
        img = Image.open(imgname)
        brightness = ImageEnhance.Brightness(img)
        img2 = brightness.enhance(factor)
        img2.save(output_dir + os.sep + imgname)
        print('split %s enhance down...' % imgname)


if __name__ == '__main__':
    factor = float(input('input enhance factor:'))
    bright_enhance(factor)
