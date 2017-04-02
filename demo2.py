import sys, pickle, time, os
sys.path.append('./lbc')
import lbc

item = '''[0] training samples
[1] image encode
[2] image decode
enter: '''

chs = input(item)
chs = int(chs)

blksize = 8
imgsize = 512
k = 5
img_name = 'testIMG/0151.bmp'
sampleDIR = 'sample/'
bksetDIR = 'blocks/'

if chs==0:
    # training
    kset = k
    vqds  = lbc.train.vq_train(sampleDIR, bksetDIR, kset, imgsize, blksize, gen_set=False, sav=True)

elif chs==1:
    # encoding
    # encode_begin = time.perf_counter()
    encode_begin = time.perf_counter()
    with open('vqdict.pkl','rb') as f:
        vqds = pickle.load(f)
    kset = [k]*len(vqds)
    hexcode = lbc.encode.encode(img_name, vqds, kset, imgsize, blksize, show=False)
    encode_end = time.perf_counter()
    consume_t = encode_end - encode_begin
    print('image encoding consume time: %.3fms'%(consume_t*1000))


elif chs==2:
    # decoding
    decode_begin = time.perf_counter()
    with open('vqdict.pkl','rb') as f:
        vqds = pickle.load(f)
    kset = [k]*len(vqds)
    hexfile = img_name[:-3] + 'hex'
    bitstream  = lbc.decode.hex2bit(hexfile)
    img = lbc.decode.decode(bitstream, vqds, kset, imgsize, blksize)
    decode_end = time.perf_counter()
    consume_t = decode_end - decode_begin
    print('image decoding consume time: %.3fms'%(consume_t*1000))
    # lbc.decode.img_show_save(img, hexfile, show=True, sav=True)
