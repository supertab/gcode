from PIL import Image
import os, glob

# 说明: 对指定目录下，指定格式图片进行批量转换
# 功能: 转 jpg, jpg2000, bmp
# 过程: 进入指定目录, 选择图像, 选择转换格式, 设置压缩质量, 转换图片
# 输出: 转换后的图像, 不覆盖原图, 保存在原图所在目录下的

convert2jpg = lambda img, q, outdir: img.save(outdir+os.path.splitext(img.filename)[0]+'_'+str(q)+'.jpg', quality=q)
convert2j2k = lambda img, q, outdir: img.save(outdir+os.path.splitext(img.filename)[0]+'_'\
                                      +str(q)+'.j2k', quality_layers=[q], quality_mode="rates")
convert2bmp = lambda img, outdir: img.save(outdir+os.path.splitext(img.filename)[0]+'.bmp')


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
图片格式转换-批量
********************'''
print(banner)
changeDir()

srcFMT = input('输入原图片格式[bmp]: ')
if len(srcFMT)==0: srcFMT='.bmp'
if '.' not in srcFMT: srcFMT='.'+srcFMT
srcIMGs = glob.glob('*'+srcFMT)

toIMG=['.jpg', '.j2k', '.bmp']
tip = '''----- 功能 -----
[1] 转 jpg
[2] 转 jpeg2000
[3] 转 bmp
[4] 退出程序
输入数字(1-4)[1]: '''
choose = input(tip)
# 默认转换格式
if len(choose)==0: choose='1'
# 输入检测
if not choose.isdigit() or choose not in list('123'): exit()
# 默认保存目录
desDIR = input('输入图片保存目录[%sDIR]: '%toIMG[int(choose)-1][1:])
if len(desDIR)==0:
    desDIR='%sDIR'%toIMG[int(choose)-1][1:]+ os.path.sep
else:
    desDIR+=os.path.sep+'%sDIR'%toIMG[int(choose)-1][1:] + os.path.sep

try:
    os.listdir(desDIR)
except:
    os.mkdir(desDIR)
# 开始转换
if choose == '1':
    qua = input('设置压缩质量(1-95)[80]: ')
    qua = 80 if len(qua)==0 else int(qua)
    for im in srcIMGs:
        convert2jpg(Image.open(im), qua, desDIR)
        print('convert %s to %s'%(im, os.path.splitext(im)[0]+toIMG[int(choose)-1]))
elif choose=='2':
    rate = input('设置压缩倍数(2-100)[5]: ')
    rate = 5 if len(rate)==0 else int(rate)
    for im in srcIMGs:
        convert2j2k(Image.open(im), rate, desDIR)
        print('convert %s to %s'%(im, os.path.splitext(im)[0]+toIMG[int(choose)-1]))
elif choose=='3':
    for im in srcIMGs:
        convert2bmp(Image.open(im), desDIR)
        print('convert %s to %s'%(im, os.path.splitext(im)[0]+toIMG[int(choose)-1]))
print('转换完毕...')
os.system('pause')



