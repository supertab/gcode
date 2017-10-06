import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

im1 = np.array(Image.open('01bmp.bmp'))
im2 = np.array(Image.open('02cbc.bmp'))
im3 = np.array(Image.open('03cbcmp.bmp'))
im4 = np.array(Image.open('04j2k.bmp'))

f, ((a1, a2), (a3, a4)) = plt.subplots(2, 2)
f.subplots_adjust(wspace=-0.2)
plt.gray()
a1.imshow(im1)
a1.axis('off')
a1.set_title('BMP', fontsize=10)
a2.imshow(im2)
a2.axis('off')
a2.set_title('CBC', fontsize=10)
a3.imshow(im3)
a3.axis('off')
a3.set_title('CBC-MP', fontsize=10)
a4.imshow(im4)
a4.axis('off')
a4.set_title('J2K', fontsize=10)
plt.show()
