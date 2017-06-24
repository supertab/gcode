import time
from algo import lbc 

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
    k=4
    srcDIR = 'D:\\gcode\\testIMG\\stdIMG_512\\crop\\'
    vqdpath = 'D:\\gcode\\vqdict_4.pkl'
    imgpath = 'D:\\gcode\\testIMG\\stdIMG_512\\crop\\camera_512.bmp'
    lbcpath = 'D:\\gcode\\testIMG\\stdIMG_512\\crop\\output\\camera_512.lbc'

    if chs==0:
        # training
        lbc.train(srcDIR, k, blksize,'vqdict_%d.pkl'%k)

    elif chs==1:
        # encoding
        # encode_begin = time.perf_counter()
        encode_begin = time.perf_counter()
        lbc.encode(imgpath, vqdpath, blksize, k)
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
