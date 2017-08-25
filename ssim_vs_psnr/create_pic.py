import sys
sys.path.append('../tools')
import meanshift
imgname = './testpic/lenna.bmp'
meanshift.meanshift(imgname, 8)
