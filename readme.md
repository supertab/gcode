# kmeans test
- **gen_random_dot**
  - init_centroid: initialize k centroid
    way1: random choose dot from dot set
    way2: lbg design algorithm
  - data: gen_random_dot
    在给定范围内，根据r计算符合的中心点，中心点的选值固定形成k个以随机点为中心，r为半径的小区域，小区域中随机投入n个点
  - dot_plot: plot dot
    format: np.array([x1,y1], [x2,y2],...)

- **keams**
  compare with scipy.cluster.vq.kmeans2
![result](http://upload-images.jianshu.io/upload_images/3022282-91d5f14af293cd28.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

`x`: initial dot
`red dot`: my_kmeans result
`triangle`: scipy library kmeans result
