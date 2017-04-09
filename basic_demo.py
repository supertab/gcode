import sys, pickle, time, os
sys.path.append('./lbc')
import lbc, bits

item = '''[0] training samples
[1] image encode
[2] image decode
enter: '''

chs = input(item)
chs = int(chs)

blksize = 8
imgsize = 1024
k = 7
img_name = 'testIMG/1904.bmp'
sampleDIR = 'sample/'
bksetDIR = 'blocks/'

if chs==0:
    # training
    # if vqd is existed, show warning message
    conti = 'Y'
    if 'vqdict.pkl' in os.listdir():
        conti = input('the vqdict.pkl is existed, continue?(Y/[N]): ')
    if conti in ['Y', 'y']:
        kset = k
        vqds  = lbc.train.vq_train(sampleDIR, bksetDIR, kset, imgsize, blksize, gen_set=True, sav=True, allin=True)

elif chs==1:
    # encoding
    # encode_begin = time.perf_counter()
    with open('vqdict.pkl','rb') as f:
        vqds = pickle.load(f)
    kset = [k]*len(vqds)
    encode_begin = time.perf_counter()
    lbc.encode.encode(img_name, vqds, kset, imgsize, blksize, show=False, sav2hex=False)
    encode_end = time.perf_counter()
    consume_t = encode_end - encode_begin
    print('image encoding consume time: %.3fms'%(consume_t*1000))

elif chs==2:
    # decoding
    decode_begin = time.perf_counter()
    with open('vqdict.pkl','rb') as f:
        vqds = pickle.load(f)
    kset = [k]*len(vqds)
    lbcfile = img_name[:-4] + '.lbc'
    #bitstream  = lbc.decode.hex2bit(hexfile)
    bitstream = bits.read(lbcfile)
    img = lbc.decode.decode(bitstream, vqds, kset, imgsize, blksize)
    decode_end = time.perf_counter()
    consume_t = decode_end - decode_begin
    print('image decoding consume time: %.3fms'%(consume_t*1000))
    lbc.decode.img_show_save(img, lbcfile, show=False, sav=True)
