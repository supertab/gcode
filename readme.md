# 匹配追踪算法原理
- **[匹配追踪算法（MatchingPursuit）原理](http://www.jianshu.com/p/411681f26848)**

# 步骤
- **step 1-初始化：**生成/选择字典库 D，并对其中的原子做归一化处理 norm(D)，常用的字典库有：[DCT](https://en.wikipedia.org/wiki/Discrete_cosine_transform)，[Gabor](http://ieeexplore.ieee.org/document/258082/?reload=true) ，初始化信号残差 r = s（s为输入信号）
- **step 2-投影：**将信号残差 r 与 D 中的每个原子 v<sub>i</sub> 做内积 p<sub>i</sub>=s<sup>T</sup>v<sub>i</sub>，记录最大的投影长度 p<sub>max</sub>，和它所对应的原子的索引 i
- **step 3-更新：**更新残差 r = s - p*v<sub>i</sub>
- **step 4-判断：**当到达设定的迭代次数，或则残差小于设定的阈值的时候停止，否则继续 step2，step3

# 函数说明
- **MP：**使用固定的迭代次数作为收敛条件
- **MPerr：**使用迭代次数和残差阈值作为收敛条件
- **demo：** 使用mp重构图片，对图像做分块处理，分成8x8小块，块与块之间重叠7个像素

# 运行结果
![reconstruct lena](https://im1.shutterfly.com/ng/services/mediarender/THISLIFE/021009631031/media/122852445820/small/1490488194/enhance)
