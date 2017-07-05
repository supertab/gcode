import numpy as np
import matplotlib.pyplot as plt
class Signal:
# 信号类，输出长度为N的信号
    def __init__(self, N=1024):
        self.N = N
        
    def triangle(self, amplitude=1, frequency=2):
        ''' 生成三角波 '''
        # amaplitde = k*T/4
        count = frequency
        T = 1/frequency
        k = 4*amplitude/T
        interval = T/(self.N/count)
        x=[interval*i for i in range(self.N)]
        x = np.array(x)
        t2x= x[self.N//count] # 一个T内的取样点数，换算成了时间间隔
        y = []
        for t in  x:
            t = t%t2x             
            if 0<=t<= T/4:
                y.append(k*t)
            elif T/4<t<=3*T/4:
                y.append(-k*t+k*T/2)
            else:
                y.append(k*(t-T))
        return x, y
    
    def white_noise(self, amplitude=5):
        # generate noise
        t = np.linspace(0,1, self.N)
        noise=[amplitude*np.random.random() for x in np.arange(self.N//2)]
        noise_neg=[-amplitude*np.random.random() for x in np.arange(self.N//2)]
        noise.extend(noise_neg)
        np.random.shuffle(noise)
        print('噪声最大值:', max(noise))
        return t, noise
    
    def recangle(self):
        pass

    def sine_db(self, fbeg=1, fend=50, pbeg=-np.pi, pend=np.pi):
        ''' 生成长度为N的正弦函数原子库 '''
        x = np.linspace(0,1,self.N)
        fre_depart = 0.2 #频率分辨率
        fre_count = (fend-fbeg)// fre_depart
        pha_depart = 0.2 #相位分辨率
        pha_count = (pend-pbeg)// pha_depart
        y_db = []
        for fre in np.linspace(fbeg, fend, fre_count):
            for pha in np.linspace(pbeg, pend, pha_count):
                y = np.sin(2*np.pi*fre*x + pha)
                y_db.append( y/np.sqrt( np.sum(y*y))) #对库中原子单位化
        # with open('db_sine_unit.pkl','wb') as f:
            # pickle.dump(y_db, f)
        # f.close()
        return x, y_db

if __name__ == "__main__":
    t, sine_lib = Signal(1024).sine_db()
    # t, tri_wave = Signal().triangle(2, 3)
    # t, noise = Signal().white_noise(3)
    for y in sine_lib:
        plt.plot(t, y)
    plt.show()