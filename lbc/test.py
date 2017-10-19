import numpy as np
import cbc
import pickle

with open('imgs_900', 'rb') as f:
    imgs = pickle.load(f)

a = [];c = []
for blks in imgs:
    centers, clusters = cbc.my_dbscan(blks, 45, 68)
    a.append(centers)
    c.append(clusters)


print('fk...')

'''
imgdir = r'E:\A_IronSample\分类测试\冷轧带钢\init\train_set'
imgsize = (128, 128)
blksize = 8
lim_h = imgsize[0] - blksize + 1
lim_w = imgsize[1] - blksize + 1

blksets = []
imglist = cbc.read_all(imgdir)
for py in range(0, lim_h, blksize):
    for px in range(0, lim_w, blksize):
        blkset = cbc.img_split(imglist, (py, px), blksize)
        blksets.append(blkset)
        print('pos:', (py, px), 'done...')

with open('imgs_%d.pkl'%imglist.shape[0], 'wb') as f:
    pickle.dump(np.array(blksets), f)
print('done...')

'''