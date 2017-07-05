import numpy as np 
from PIL import Image
from os import path
import sys

def meanshift(imgName, factor=25):
    names = path.splitext(imgName)
    im = Image.open(imgName)
    if im.mode !='L':
        im = im.convert('L')
    ima = np.array(im).astype(np.int32)
    ima += factor
    ima[ima>255]=255
    ima[ima<0]=0
    ima = ima.astype(np.uint8)
    factor = '+'+str(factor) if factor>=0 else str(factor)
    Image.fromarray(ima).save(names[0]+factor+names[1])

def main():
    imgName = sys.argv[1]
    meanshift(imgName, 25)

if __name__ == '__main__':
    main()
