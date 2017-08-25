import os
from PIL import Image


def type_in(img_list):
    # imgType = input('输入图片的格式，按回车默认保存: ')
    imgType = input('enter image type [keep]:')
    if len(imgType) == 0:
        imgType = os.path.splitext(img_list[0])[1]
    if '.' not in imgType:
        imgType = '.' + imgType
    return imgType


def get_img_list(desPath):
    print('直接按回车表示按默认参数...')
    while True:
        # srcPath = input('输入源图片所在路径，默认为当前目录： ')
        srcPath = input('enter the path of image [./]:')
        if len(srcPath) == 0:
            srcPath = './'
            break
        try:
            os.chdir(srcPath)  # change to srcPath
            break
        except:
            # print('该目录不存在...')
            print('fail to get in this dir...')
            continue
    # get image list, then make the dir for convert
    srcDir = os.listdir()
    img_list = [i for i in srcDir if i[-3:] in ['bmp', 'jpg', 'png']]
    if len(img_list) == 0:
        raise Exception('no image in the dir...')
    if desPath[:-1] not in srcDir:
        os.mkdir(desPath)  # mkdir in srcDir
    return img_list, srcPath


def rgb2gray(desPath='gray_img/'):
    img_list, srcPath = get_img_list(desPath)
    imgType = type_in(img_list)
    for each_im in img_list:
        im = Image.open(each_im)
        if im.mode in ['RGB', 'RGBA']:
            im = im.convert('L')
        img_name = desPath + os.path.splitext(each_im)[0] + imgType
        im.save(img_name)
        print('convert: %s to %s' % (each_im, img_name))
    # print('图片保存目录为%s'%(srcPath+'\\'+desPath[:-1]))
    print('image saved in: %s' % (srcPath + '\\' + desPath[:-1]))
    input()


def crop(desPath='crop/'):
    img_list, srcPath = get_img_list(desPath)
    imgType = type_in(img_list)
    img_size = Image.open(img_list[0]).size
    while True:
        area = input('size of input image：%s\nenter left, upper, right, lower [0 0 %d %d]: ' % (
            str(img_size), img_size[0], img_size[1]))
        if len(area) == 0:
            area = '0 0 %d %d' % (img_size[0], img_size[1])
        area = area.split()
        (left, upper, right, lower) = tuple([int(i) for i in area])
        if left < right <= img_size[0]:
            if upper < lower <= img_size[1]:
                break
    for each_im in img_list:
        im = Image.open(each_im)
        im = im.crop((left, upper, right, lower))
        img_name = desPath + os.path.splitext(each_im)[0] + imgType
        im.save(img_name)
        print('convert: %s to %s' % (each_im, img_name))
    input()


def resize(desPath='resize/'):
    img_list, srcPath = get_img_list(desPath)
    imgType = type_in(img_list)
    _size = Image.open(img_list[0]).size
    im_info = 'the image size is %s' % str(_size)
    scal = input('%s, enter the scale[1.0]: ' % im_info)
    if len(scal) == 0:
        scal = 1.0
    else:
        scal = float(scal)
    imgsize = (int(_size[0] * scal), int(_size[1] * scal))
    for each_im in img_list:
        im = Image.open(each_im)
        if scal != 1.0:
            im = im.resize(imgsize)
        img_name = desPath + os.path.splitext(each_im)[0] + imgType
        im.save(img_name, quality=90)
        print('convert: %s to %s' % (each_im, img_name))

    print('image saved in: %s' % (srcPath + '\\' + desPath[:-1]))
    # print('图片保存目录为%s'%(srcPath+'\\'+desPath[:-1]))
    print('new image is %s' % str(imgsize))


def changeDir():
    while True:
        # 切换目录
        workdir = input('输入源图片文件所在目录: ')
        try:
            if len(workdir):
                os.chdir(workdir)
            break
        except:
            print('目录有误...')
    print('工作目录为: %s' % os.getcwd())


changeDir()
func = [rgb2gray, crop, resize]
while True:
    title = '''
-------------------------
IMAGE Batch Tool v1
-------------------------
* [1] rgb2gray
* [2] crop
* [3] resize
* [q] quit
-------------------------
输入:'''
    chs = input(title)
    if chs == 'q':
        break
    if chs.isdigit() and int(chs) in range(1, len(func) + 1):
        func[int(chs) - 1]()
        break
