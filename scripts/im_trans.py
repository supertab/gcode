import numpy as np

def im2col(img, blockshape, stepsize=1):
    w, h = img.shape
    wlim = w - blockshape[0]+1
    hlim = w - blockshape[1]+1
    tmp = []
    for x in range(wlim):
        for y in range(hlim):
            tmp.append( img[x:x+8, y:y+8].reshape(64))
    return np.asarray(tmp).T

def col2im(im_size,blocks, bb):
    # 重构图像
    # genrate pos of every pixes by Image and blocks
    x_lim = im_size[0] - bb
    y_lim = im_size[1] - bb
    # padding
    recon_image = np.zeros(im_size)
    weight = np.zeros(im_size)
    col= 0
    for x_pos in range(x_lim+1):
        # for y_pos,each_block in zip(range(y_lim+1), blocks): # lead Error
        for y_pos in range(y_lim+1):
            # padding block
            recon_image[x_pos:x_pos+bb, y_pos:y_pos+bb] += blocks[:,col].reshape((bb,bb))
            # record weight
            weight[x_pos:x_pos+bb, y_pos:y_pos+bb] += np.ones((bb,bb))
            col += 1
    return recon_image / weight