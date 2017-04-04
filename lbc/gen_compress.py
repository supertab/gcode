import PIL.Image as Image
from os import path

# compress as jpg
def bmp2jpg(pic, des_size, decomp=True):
    '''
    parameters:
    pic: path of picture to compress, only support bmp.
    des_size: output image's size, unit is kb
    return:
    pic_jpg: Image object
    size_out: picture's size, Kb
    '''
    des_size = des_size*1024
    im_bmp = Image.open(pic)
    pic_jpg = path.splitext(pic)[0]+'.jpg'
    # binary search: target: des_size, compare: jpg save size
    low = 1
    high = 95
    while low < high:
        mid = low + (high-low)//2
        im_bmp.save(pic_jpg, quality= mid)
        get_size = path.getsize(pic_jpg)
        # print('%.3f'%(get_size/1024), mid)
        if des_size > get_size:
            low = mid + 1
        elif des_size < get_size:
            high = mid - 1
        else: 
            low = high = mid
    im_bmp.save(pic_jpg, quality=mid+1)
    size_out = path.getsize(pic_jpg)/1024
    im_jpg = Image.open(pic_jpg)
    if decomp:
        im_jpg.save(pic_jpg.replace('.', '_')+'.bmp')
    return im_jpg, size_out 

def bmp2j2k(pic, des_size, decomp=True):
    size_in = path.getsize(pic)
    r = size_in//( des_size*1024)
    im_bmp = Image.open(pic)
    pic_j2k = path.splitext(pic)[0]+'.j2k'
    im_bmp.save(pic_j2k, quality_mode='rates', quality_layers=[r])
    size_out = path.getsize(pic_j2k)/1024
    im_j2k = Image.open(pic_j2k)
    if decomp:
        im_j2k.save(pic_j2k.replace('.', '_')+'.bmp')
    return im_j2k, size_out

if __name__ =='__main__':
    pic_in = "0006.bmp"
    im_jpg, jpgsize  = bmp2jpg(pic_in, 2)
    im_j2k, j2ksize = bmp2j2k(pic_in, 2)
    print('sdf')

