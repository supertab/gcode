import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

im1 = np.array(Image.open('cbc.bmp').convert('L'))
im2 = np.array(Image.open('cbcmp.bmp').convert('L'))
im3 = np.array(Image.open('jpg.bmp').convert('L'))
im4 = np.array(Image.open('j2k.bmp').convert('L'))

f, ((a1, a2), (a3, a4)) = plt.subplots(2, 2)
f.subplots_adjust(wspace=-0.21)
plt.gray()
a1.imshow(im1)
a1.axis('off')
a1.set_title('CBC', fontsize=10)
a2.imshow(im2)
a2.axis('off')
a2.set_title('CBC-MP', fontsize=10)
a3.imshow(im3)
a3.axis('off')
a3.set_title('JPG', fontsize=10)
a4.imshow(im4)
a4.axis('off')
a4.set_title('J2K', fontsize=10)
plt.show()
