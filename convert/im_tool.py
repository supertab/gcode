import os
from PIL import Image

def type_in():
    print('直接按回车表示按默认参数...')
    imgType = input('输入图片的格式，按回车默认保存为 jpg: ')
    if len(imgType)==0:
        imgType='jpg'
    while True:
        srcPath = input('输入源图片所在路径，默认为当前目录： ')
        if len(srcPath)==0:
            srcPath = './'
            break
        try:
            os.chdir(srcPath) # change to srcPath
            break
        except:
            print('该目录不存在...')
            continue
    return imgType, srcPath

def get_img_list(desPath):
    srcDir = os.listdir() 
    img_list = [ i for i in srcDir if i[-3:] in ['bmp', 'jpg', 'png']]
    if len(img_list)==0:
        print('没有发现图片...')
        input()
        return None
    if desPath[:-1] not in srcDir:
        os.mkdir(desPath) # mkdir in srcDir
    return img_list

def rgb2gray(desPath='gray_img/'):
    imgType, srcPath = type_in()
    img_list = get_img_list(desPath)
    if img_list == None:
        return
        
    for each_im in img_list:
        im = Image.open(each_im)
        if im.mode == 'RGB':
            im = im.convert('L')
        img_name = desPath+each_im[:-3]+imgType
        im.save(img_name)
        print('convert: %s to %s'%(each_im, img_name))
    print('图片保存目录为%s'%(srcPath+'\\'+desPath[:-1]))
    input()    

def crop(desPath='crop/'):
    imgType, srcPath = type_in()
    img_list = get_img_list(desPath)
    if img_list == None:
        return
    img_size = Image.open(img_list[0]).size
    while True:
        area = input('图片尺寸为：%s\n\
               输入 左上点 和 右下点 的坐标(格式 0 0 128 128）: '%str(img_size))
        area = area.split()
        (left, upper, right, lower) = tuple([int(i) for i in area])
        if left < right < img_size[0]:
            if upper < lower < img_size[1]:
                break
    for each_im in img_list:
        im = Image.open(each_im)
        im = im.crop((left, upper, right, lower))
        img_name = desPath+each_im[:-3]+imgType
        im.save(img_name)
        print('convert: %s to %s'%(each_im, img_name))
    print('图片保存目录为%s'%(srcPath+'\\'+desPath[:-1]))
    input()   

if __name__ == '__main__':
    func = [rgb2gray, crop]
    item ='''【0】RGB转灰度图
【1】裁剪图像
输入：'''
    chs = input(item)
    func[int(chs)]()

