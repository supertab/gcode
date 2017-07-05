# 程序列表
- [mp](https://github.com/supertab/gcode/tree/mp): 
匹配追踪算法（MatchingPursuit），一种重构算法
- [kmeansVQ](https://github.com/supertab/gcode/tree/kmeansVQ)
kmeans聚类算法的实现与应用-rgb图片压缩
- [lowbitCompress](https://github.com/supertab/gcode/tree/lowbitCompress)
专用，低比特压缩算法

# 目录结构
```
|--lbc: 低比特压缩算法
|--mp: 匹配追踪算法
|  |--image_recon: 图像重构实验，比较DCT和Gabor字典的重构效果
|  |--signal: 正弦信号重构实验，比较sin字典和Gabor字典
|--scripts: 公用程序
|--testIMG: 公用测试图片
|  |--AlphaIMG: 256x256 字母图片
|  |--ironIMG: 
|  |--stdIMG: 128, 256, 512标准测试图片
|--tools: 图片转换，psnr，ssim工具
```