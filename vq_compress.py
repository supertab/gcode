import numpy as np
import scipy.cluster.vq as vq
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

f_name = 'lena200.bmp'
img =np.array( Image.open(f_name), dtype='float64')/255
data = vq_encode(f_name)
# recover photo
im = vq_decode( data)
compr_img = Image.fromarray((im*255).astype(np.uint8))
compr_img.save('reconstruct_lena.bmp')

#plot
# fig = plt.figure(1)
# ax1 = fig.add_subplot(121)
# ax1.imshow(img)
# plt.title('initial image')
# ax2 = fig.add_subplot(122)
# ax2.imshow(compr_img)
# plt.title('vq compress image')
# plt.show()

#save compressed data
# save_data(data, 'lena200')



