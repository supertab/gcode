import pickle,os
import numpy as np 
import matplotlib.pyplot as plt 
# import matplotlib as mpl
# mpl.rc('font', family='SimHei')

n=512
t = np.linspace(0,1,n)
y1=3*np.sin(2*np.pi*2*t+np.pi/3)
y2=np.sin(2*np.pi*5*t+np.pi/6)
y3 = np.sin(2*np.pi*23*t)
y = y1+y2+y3

with open('output%ssignal_sin%s.pkl'%(os.sep,str(n)),'rb') as f:
    #y_pos, coefs, re_y, recon1
    y_pos, coefs, re_y, recon1 = pickle.load(f)
with open('output%ssignal_gabor%s.pkl'%(os.sep,str(n)),'rb') as f:
    y_pos2, coefs2, re_y2, recon2 = pickle.load(f)


fig, ((ax0, ax1), (ax2, ax3))=  plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(8,8))
fig.subplots_adjust(wspace=0.1, hspace=0.2)
ax0.set(title=r'$y1+y2+y3$', xlabel='(a)')
ax0.plot(y, label='signal')
ax0.plot(recon1, '-.', label='recon-Sin')
ax0.plot(recon2, '--', label='recon-Gabor')
ax0.legend()
ax1.set(title=r'$y1=3sin(4\pi t+\pi/3)$', xlabel='(b)')
ax1.plot(y1, label='signal')
ax1.plot(re_y[0], '-.', label='recon-Sin')
ax1.plot(re_y2[0], '--', label='recon-Gabor')
# ax1.legend()
ax2.set(title=r'$y2=sin(10\pi t+\pi/6)$', xlabel='(c)')
ax2.plot(y2,label='signal')
ax2.plot(re_y[1], '-.',label='recon-Sin')
ax2.plot(re_y2[1], '--',label='recon-Gabor')
# ax2.legend()
ax3.set(title=r'$y3=sin(46\pi t)$', xlabel='(d)')
ax3.plot(y3,label='signal')
ax3.plot(re_y[2], '-.',label='recon-Sin')
ax3.plot(re_y2[2], '--',label='recon-Gabor')
# ax3.legend()
# plt.show()
