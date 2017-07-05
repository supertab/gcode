import numpy as np
import matplotlib.pyplot as plt
import mp_dicts, mp2
import pickle
import matplotlib as mpl
err1 = [32.13, 22.23, 3.13, 2, 1.93, 1.52, 1.2,1.0, 0.623, 0.512, 0.4, 0.32, 0.28, 0.2, 0.1]
mpl.rc('font', family='SimHei')
sin_line, = plt.plot(list(range(1,16)),err1, '-o', label='Sin原子库')
# plt.axis([1, 16, 0, 35])
# gabor_line, = plt.plot(err2, '-^', label='Gabor原子库')
# plt.legend(handles=[sin_line, gabor_line])
plt.xlabel('迭代次数'), plt.ylabel('重构误差')
plt.show()