# 目录说明
- `lbc/`    源码目录
- `sample/` 放置训练样本
- `blocks/` 放置分块后的样本，样本块由 train 函数生成
- `testIMG/` 放置测试图片(测试图片只能为 bmp 格式）， encode 从此目录读取图片，生成的压缩文件保证在此目录>中，decode 还原的图片也保存在此目录中
- `convert/` 对图片进行灰度变换，裁剪，改变尺寸（批量）
- 训练得到的 VQ 字典保存在当前目录，名为 vqdict.pkl

# 实现结果
![](http://upload-images.jianshu.io/upload_images/3022282-d20fc4c72734e35d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
