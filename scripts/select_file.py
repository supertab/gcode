#!/usr/bin/python3
# encoding:utf-8
import os
def select_file(*args, destdir='./'):
    try: os.listdir(destdir)
    except: return -1
    suffix=['.cc', '.py', '.txt', '.md']
    if len(args): suffix.extend(args)
    flist = [f for f in os.listdir(destdir) if os.path.splitext(f)[1].lower() in suffix]
    flist.sort()
    files = ''.join([ str(idx+1) + ' ' + f + '\n' for (idx, f) in enumerate(flist)])
    fidx = input('%s选择文件(输入数字)：'%files)
    if fidx.isdigit() and 0<int(fidx)<=len(flist):
        return flist[int(fidx)-1]
    else: return -1

if __name__ == '__main__':
    dest = './'
    dir_in = input('enter target dir([./]): ')
    if dir_in: dest = dir_in
    formt = ['.bmp', '.jpg']
    print(select_file(*formt, destdir=dest))
