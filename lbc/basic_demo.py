import time, os, glob
import lbc 

'''
在当前目录中保存聚类字典：.pkl 文件
图像压缩文件和解压缩文件在图像路径目录下的 output 子目录中
'''

item = '''[0] training samples
[1] image encode
[2] image decode
[q] quit
enter: '''

mp = True if input('use mp(Y/N)?: ') in 'yY' else False

while True:
    chs = input(item)
    if chs=='q': break
    chs = int(chs)
    blksize=8
    k_list=[3, 4, 5, 6, 7]
    # srcDIR = 'E:\\WZ\\gcode\\testIMG\\stdIMG\\256'
    # vqdpath = '.\\vqdict_%d.pkl'%k
    # imgpath = '..\\testIMG\\stdIMG\\256\\lenna.bmp'
    # lbcpath = '..\\testIMG\\stdIMG\\256\\output\\lenna.lbc'
    srcDIR = 'E:\\WZ\\金属样本\\西南铝\\sams\\left'
    imgidx = '0064'
    imgpath = 'E:\\WZ\\金属样本\\西南铝\\sams\\test_left\\%s.bmp'%imgidx
    lbcpath = 'E:\\WZ\\金属样本\\西南铝\\sams\\test_left\\output\\%s.lbc'%imgidx

    for k in k_list:
        vqdpath = '.\\vqdict_%d.pkl'%k
        vqdname = os.path.split(vqdpath)[1]
        kset = lbc.gen_kset(k, srcDIR, blksize)
        if chs==0:
            # training
            lbc.train(srcDIR, kset, blksize)
            os.rename('vqdict.pkl',vqdname)
        elif chs==1:
            # encoding
            # encode_begin = time.perf_counter()
            encode_begin = time.perf_counter()
            imglbc, lbcsize = lbc.encode(imgpath,vqdpath, blksize, mp=mp, mode='rmse')
            encode_end = time.perf_counter()
            consume_t = encode_end - encode_begin
            print('image encoding consume time: %.3fms, lbcsize: %.3f Kb\n'%(consume_t*1000, lbcsize/1024))
            os.rename(lbcpath, lbcpath.replace(imgidx, imgidx+'_%s'%k))

        elif chs==2:
            # decoding
            decode_begin = time.perf_counter()
            lbc.decode(lbcpath.replace(imgidx, imgidx+'_%s'%k), vqdpath, blksize)
            decode_end = time.perf_counter()
            consume_t = decode_end - decode_begin
            print('image decoding consume time: %.3fms\n'%(consume_t*1000))
            # os.rename(savdimg, savdimg.replace('lbc','lbc%s'%k))
