from PIL import Image
import os

# 说明: 单幅图像格式转换, bmp 转 jpg, jpg2000
# 过程: 进入指定目录, 选择图像, 选择转换格式, 设置压缩质量, 转换图片
# 输出: 转换后的图像, 不覆盖原图, 保存在原图所在目录


def select_file(*args):
    suffix=[]
    if len(args): suffix.extend(args)
    flist = [f for f in os.listdir('.') if os.path.splitext(f)[1].lower() in suffix] 
    files = ''.join([ str(idx+1) + ' ' + f + '\n' for (idx, f) in enumerate(flist)])
    print('----- 当前目录中的文件 -----\n%s'%files, end='')
    while True:
        fidx = input('选择图片(输入数字)：')
        if fidx.isdigit() and 0<int(fidx)<=len(flist):
            return flist[int(fidx)-1]
    

convert2jpg = lambda img, q: img.save(os.path.splitext(img.filename)[0]+'_'+str(q)+'.jpg', quality=q)
convert2j2k = lambda img, q: img.save(os.path.splitext(img.filename)[0]+'_'\
                                      +str(q)+'.j2k', quality_layers=[q], quality_mode="rates")
convert2bmp = lambda img: img.save(os.path.splitext(img.filename)[0]+'.bmp')


def changeDir():
    while True:
        # 切换目录
        workdir = input('输入源图片文件所在目录: ')
        try:
            if len(workdir): os.chdir(workdir)
            print('工作目录为: %s'%os.getcwd())
            break
        except:
            print('目录有误...')

banner = '''********************
图片格式转换-单张
********************
'''
print(banner)
changeDir()
tip = '''----- 功能 -----
[1] 转 jpg
[2] 转 jpeg2000
[3] 转 bmp
[4] 退出程序
输入数字(1-4)[1]: '''
while True:
    choose = input(tip)
    if len(choose)==0: choose='1'
    if choose.isdigit() and choose in list('123'):
        img_in = select_file('.bmp', '.jpg', '.j2k')
        if img_in==None: continue
        im = Image.open(img_in)
        if choose == '1':
            qua = input('设置压缩质量(1-95)[80]: ')
            qua = 80 if len(qua)==0 else int(qua)
            convert2jpg(im, qua)
        elif choose=='2':
            rate = input('设置压缩倍数(2-100)[5]: ')
            rate = 5 if len(rate)==0 else int(rate)
            convert2j2k(im, rate)
        elif choose=='3': convert2bmp(im)
        print('转换完毕...')
    else: break



