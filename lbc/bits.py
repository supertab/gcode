import ctypes
# str2bits
def write(bs_in, fname):
    dll = ctypes.cdll.LoadLibrary
    bitstream = dll('lbc/bitstream.so')
    bs_in = bs_in.encode('utf-8')
    fname = fname[:-4]+'.lbc'
    fname = fname.encode('utf-8')
    bitstream.write(bs_in, fname)

# parse hex stream
def hex2bit(fin):
    if '.' in fin:
        with open(fin) as f:
            hexstr = f.read()
    else: hexstr=fin
    bitstream = ''
    for each_hex in hexstr:
        # hex to binary
        bits = bin(int(each_hex, 16))[2:]
        # put in bit jar
        bits = '0'*(4-len(bits)) + bits
        bitstream += bits
    return bitstream

# read from file
def read(fn):
    with open(fn, "rb") as f:
        d_hex = f.read().hex()
    return hex2bit(d_hex)

if __name__=='__main__':
    '''
    the len of bit stream should be multiples of 8bits
    '''
    #a = '011111101110001100100000'
    a = '1'*24
    _a = a.encode('utf-8')
    fn = "str.bits"
    _fn = fn.encode('utf-8')
    write(_a, _fn)
    print(read(fn))
