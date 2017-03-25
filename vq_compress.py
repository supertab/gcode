import numpy as np
import scipy.cluster.vq as vq
from kmeans import *
from PIL import Image
import matplotlib.pyplot as plt


# normR = whiten(rand_dot) # normlize, not mean normlize: each feature/standard deviation


'''
# kmeans
# input vector, the column is each feature
# return centriod and distortion
# kmeans2
# input some as kmeans, it can change the initial centriod
# return centriod and the label shows which group is the sample in
'''
def vq_encode(img, k=127):
    if isinstance(img, str):
        img_data =np.array( Image.open(img), dtype='float64')/255
    elif isinstance(img, np.ndarray):
        img_data = img
    else:
        print('encode err...')
        return
    weigth, hight, channal = img_data.shape
    img_seq = img_data.reshape((weigth*hight, channal))
    (code_book, label_k2) = vq.kmeans2(img_seq, k )
    # (code_book, label_k2) = my_kmeans(img_seq, k )
    label_k2 = label_k2.astype(np.uint8)
    pack_data = (code_book, label_k2, img_data.shape)
    # return code_book, label_k2, im_shape
    return pack_data

def vq_decode(data, fromfile=False):
    if isinstance(data, str):
        import pickle
        with open(data, 'rb') as f:
            img_data = pickle.load(f)
    elif isinstance(data, tuple):
        img_data = data
    else:
        print('decode err...')
        return
    (code_book , label, img_shape) = img_data
    im_cs = code_book[label, :]
    # return array of img
    return im_cs.reshape(img_shape)

def save_data(pack_data, fname):
    import pickle
    with open(fname+'.vq', 'wb') as f:
        pickle.dump(pack_data, f)

f_name = 'lena256.jpg'
img =np.array( Image.open(f_name), dtype='float64')/255
data2 = vq_encode(f_name, 2)
data64 = vq_encode(f_name, 64)
data100 = vq_encode(f_name, 100)
# recover photo
im2 = vq_decode( data2)
im64 = vq_decode( data64)
im100 = vq_decode( data100)
compr_img2 = Image.fromarray((im2*255).astype(np.uint8))
compr_img64 = Image.fromarray((im64*255).astype(np.uint8))
compr_img100 = Image.fromarray((im100*255).astype(np.uint8))
# compr_img.save('reconstruct_lena.bmp')

#plot
fig = plt.figure(1)
ax1 = fig.add_subplot(221)
plt.title('initial image')
ax1.imshow(img)
ax2 = fig.add_subplot(222)
plt.title('k=2')
ax2.imshow(compr_img2)
ax3 = fig.add_subplot(223)
plt.title('k=64')
ax3.imshow(compr_img64)
ax4 = fig.add_subplot(224)
plt.title('k=100')
ax4.imshow(compr_img100)
plt.show()

#save compressed data
# save_data(data, 'lena200')



