from PIL import Image
import os

# 说明: 单幅图像格式转换, bmp 转 jpg, jpg2000
# 过程: 进入指定目录, 选择图像, 选择转换格式, 设置压缩质量, 转换图片
# 输出: 转换后的图像, 不覆盖原图, 保存在原图所在目录


def select_file(*args):
    suffix=[]
    if len(args): suffix.extend(args)
    flist = [f for f in os.listdir('.') if os.path.splitext(f)[1] in suffix] 
    files = ''.join([ str(idx+1) + ' ' + f + '\n' for (idx, f) in enumerate(flist)])
    print('----- 当前目录中的文件 -----')
    fidx = input('%s选择原图(输入数字)：'%files)
    if fidx.isdigit() and 0<int(fidx)<=len(flist):
        return flist[int(fidx)-1]

convert2jpg = lambda img, q: img.save(os.path.splitext(img.filename)[0]+'_'+str(q)+'.jpg', quality=q)
convert2j2k = lambda img, q: img.save(os.path.splitext(img.filename)[0]+'_'\
                                      +str(q)+'.j2k', quality_layers=[q], quality_mode="rates")

def changeDir():
    while True:
        # 切换目录
        workdir = input('输入源图片文件所在目录: ')
        try:
            if len(workdir): os.chdir(workdir)
            break
        except:
            print('目录有误...')
        print('工作目录为: %s'%os.getcwd())

if __name__=='__main__':
    func = [convert2jpg, convert2j2k]
    changeDir()
    tip = '''----- 功能 -----
[1] 转 jpg
[2] 转 jpeg2000
输入数字[1]: '''
    while True:
        choose = input(tip)
        if len(choose)==0: choose='1'
        if choose.isdigit() and choose in '12':    
            img_in = select_file('.bmp', '.jpg', '.j2k')
            qua = input('设置压缩质量或者倍数(1-95)[50]: ')
            if len(qua)==0: qua=50
            else: qua = int(qua)
            im = Image.open(img_in)
            func[int(choose)-1](im, qua)
            #os.startfile('0045.jpg')
        else: break

    

