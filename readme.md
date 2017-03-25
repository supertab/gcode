# kmeans算法原理
- [简书 - kmeans原理与应用](http://www.jianshu.com/p/594e9fc5db9a)

# kmeans程序测试
- **gen_random_dot程序**
  - init_centroid: initialize k centroid
    way1: random choose dot from dot set
    way2: lbg design algorithm
  - data: gen_random_dot
    在给定范围内，根据r计算符合的中心点，中心点的选值固定形成k个以随机点为中心，r为半径的小区域，小区域中随机投入n个点
  - dot_plot: plot dot
    format: np.array([x1,y1], [x2,y2],...)

- **kmeans程序**
  - 自己实现的 kmeans 与 scipy 库函数进行对比实验
![result](http://upload-images.jianshu.io/upload_images/3022282-91d5f14af293cd28.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

     - `x`: initial dot
     - `red dot`: my_kmeans result
     - `triangle`: scipy library kmeans result

# 应用-图像压缩
- **vq_compress程序**
  - vq_encode: 编码函数，输入图像矩阵，k，输出 (centroid, label, im.shape)三元组
  - vq_decode: 解码函数，从文件或者内存中传入(centroid, label, im.shape)三元组，返回图像数组
  - save_data：将三元组以文件的形式保存，后缀为vq

![实验结果](http://upload-images.jianshu.io/upload_images/3022282-c4b0b41f39e515a0.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
