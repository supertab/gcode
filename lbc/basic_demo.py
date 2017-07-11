import time
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

while True:
    chs = input(item)
    if chs=='q': break
    chs = int(chs)
    blksize=8
    k=3
    srcDIR = 'E:\\WZ\\gcode\\testIMG\\stdIMG\\256'
    # srcDIR = 'G:\\图像库\\北海固溶二号线图像\\二号线成卷图像\\003652bmp'
    vqdpath = '.\\vqdict_%d.pkl'%k
    imgpath = '..\\testIMG\\stdIMG\\256\\lenna.bmp'
    # imgpath = 'G:\\图像库\\北海固溶二号线图像\\二号线成卷图像\\tmp\\flaw01.bmp'
    lbcpath = '..\\testIMG\\stdIMG\\256\\output\\lenna.lbc'
    # lbcpath = 'G:\\图像库\\北海固溶二号线图像\\二号线成卷图像\\tmp\\output\\flaw01.lbc'

    if chs==0:
        # training
        lbc.train(srcDIR, k, blksize,'vqdict_%d.pkl'%k)

    elif chs==1:
        # encoding
        # encode_begin = time.perf_counter()
        encode_begin = time.perf_counter()
        lbc.encode(imgpath, vqdpath, blksize, k, mode='rmse')
        encode_end = time.perf_counter()
        consume_t = encode_end - encode_begin
        print('image encoding consume time: %.3fms'%(consume_t*1000))

    elif chs==2:
        # decoding
        decode_begin = time.perf_counter()
        lbc.decode(lbcpath, vqdpath, blksize, k)
        decode_end = time.perf_counter()
        consume_t = decode_end - decode_begin
        print('image decoding consume time: %.3fms'%(consume_t*1000))
