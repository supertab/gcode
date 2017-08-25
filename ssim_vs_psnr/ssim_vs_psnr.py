import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys
sys.path.append('../scripts')
import quota

a0 = np.array(Image.open('testpic/A3.png').convert('L'))
a1 = np.array(Image.open('testpic/A3_r180.png').convert('L'))
a2 = np.array(Image.open('testpic/B.png').convert('L'))

lena0 = np.array(Image.open('testpic/lenna.bmp'))
lena1 = np.array(Image.open('testpic/lenna+8.bmp'))
lena2 = np.array(Image.open('testpic/lenna_5.jpg'))

# print psnr, ssim
print('psnr-r180:', quota.psnr(a0, a1))
print('ssim-r180:', quota.mssim(a0, a1))
print('psnr-diff:', quota.psnr(a0, a2))
print('ssim-diff:', quota.mssim(a0, a2))
print('psnr-meanshift:', quota.psnr(lena0, lena1))
print('ssim-meanshift:', quota.mssim(lena0, lena1))
print('psnr-jpg', quota.psnr(lena0, lena2))
print('ssim-jpg:', quota.mssim(lena0, lena2))

# show image
f, ((ax0, ax1, ax2), (ax3, ax4, ax5)) = plt.subplots(
    2, 3)  # figsize=(5,2.8)只能单方向（单参数）调整
f.subplots_adjust(wspace=0.1)  # 可以为负值
plt.gray()
ax0.imshow(a0)
ax0.axis('off')
ax0.set_title('Initial-A')

ax1.imshow(a1)
ax1.axis('off')
ax1.set_title('rotate180')

ax2.imshow(a2)
ax2.axis('off')
ax2.set_title('different')

ax3.imshow(lena0)
ax3.axis('off')
ax3.set_title('Initial-lenna')

ax4.imshow(lena1)
ax4.axis('off')
ax4.set_title('meanshift8')

ax5.imshow(lena2)
ax5.axis('off')
ax5.set_title('compress')
f.savefig('allimg.jpg')
# plt.show()


