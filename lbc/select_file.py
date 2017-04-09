import os
import os.path as path

def select_file(f_format, dest_dir):
    conts = [i for i in os.listdir(dest_dir) if path.splitext(i)[1].lower()==f_format ]
    conts.sort()
    files =''.join([str(idx+1) + ' ' + f + '\n' for idx, f in enumerate(conts)])
    fidx = input('%s选择文件(输入数字): '%files)
    if fidx.isdigit() and 0<int(fidx)<=len(conts):
        return conts[int(fidx)-1]
    else:
        return -1

if __name__=='__main__':
    f_format = '.bmp'
    dest_dir = 'testIMG/'
    print( select_file(f_format, dest_dir))

