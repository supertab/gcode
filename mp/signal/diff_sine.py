import sys; sys.path.append('../../scripts')
import numpy as np
import matplotlib.pyplot as plt
import mp_dicts, mp2, quota
import pickle, os
import matplotlib as mpl
# mpl.rc('font', family='SimHei')

'''
# 当n=1024时程序运行需要12G内存空间
'''

# signal
n = 256
# sin dictnary
sindb = mp_dicts.GenSin(n)
gabor = mp_dicts.GenGabor(n)
t = np.linspace(0,1,n)
y1=3*np.sin(2*np.pi*2*t+np.pi/3)
y2=np.sin(2*np.pi*5*t+np.pi/6)
y3 = np.sin(2*np.pi*23*t)
y = y1+y2+y3

sindb_recon, gabor_recon =[], []    # record the results of mp
sindb_err, gabor_err=[],[]          # record the err of initial signal and reconstruct
sindb_atoms, gabor_atoms=[], []     # the selected atom from each  iterator

for w in range(1,4):
    y_res= mp2.encode(sindb, y, maxErr=0, maxIter=w)[:,0]
    y_res2 = mp2.encode(gabor, y, maxErr=0, maxIter=w)[:,0]
    srecon = np.dot(sindb, y_res)   # reconstruct signal
    grecon = np.dot(gabor, y_res2)  
    sindb_recon.append(srecon)      # save reconsruct signal in list
    gabor_recon.append(grecon)

# save atoms in list
y_pos = y_res.nonzero()[0]
y_pos2 = y_res2.nonzero()[0]
coefs = y_res[y_pos]
coefs2 = y_res2[y_pos2]
for pos, coef in zip(y_pos, coefs):
    t = sindb.T[pos]*coef
    sindb_atoms.append(t)
for pos, coef in zip(y_pos2, coefs2):
    t = gabor.T[pos]*coef
    gabor_atoms.append(t)


##plot 重构与分解图
fig1, ((ax0, ax1), (ax2, ax3))=  plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(8,8))
fig1.subplots_adjust(wspace=0.1, hspace=0.2)
ax0.set(title=r'$y1+y2+y3$', xlabel='(a)')
ax0.plot(y, label='signal')
ax0.plot(sindb_recon[2], '-.', label='recon-Sin')
ax0.plot(gabor_recon[2], '--', label='recon-Gabor')
ax0.legend()
ax1.set(title=r'$y1=3sin(4\pi t+\pi/3)$', xlabel='(b)')
ax1.plot(y1, label='signal')
ax1.plot(sindb_atoms[0], '-.', label='recon-Sin')
ax1.plot(gabor_atoms[0], '--', label='recon-Gabor')
# ax1.legend()
ax2.set(title=r'$y2=sin(10\pi t+\pi/6)$', xlabel='(c)')
ax2.plot(y2,label='signal')
ax2.plot(sindb_atoms[1], '-.',label='recon-Sin')
ax2.plot(gabor_atoms[1], '--',label='recon-Gabor')
# ax2.legend()
ax3.set(title=r'$y3=sin(46\pi t)$', xlabel='(d)')
ax3.plot(y3,label='signal')
ax3.plot(sindb_atoms[2], '-.',label='recon-Sin')
ax3.plot(gabor_atoms[2], '--',label='recon-Gabor')


for w in range(1,16):
    y_res= mp2.encode(sindb, y, maxErr=0, maxIter=w)[:,0]
    y_res2 = mp2.encode(gabor, y, maxErr=0, maxIter=w)[:,0]
    srecon = np.dot(sindb, y_res)   # reconstruct signal
    grecon = np.dot(gabor, y_res2)  
    serr = quota.veclen(y-srecon)   # calculate the length of error vector 
    gerr = quota.veclen(y-grecon)
    sindb_err.append(serr)          # save reconstruct error in list
    gabor_err.append(gerr)

# 迭代次数和重构误差关系图
mpl.rc('font', family='SimHei')
fig2 = plt.figure()
plt.xlim(0,16)
x = list(range(1,16))
sinl, = plt.plot(x,sindb_err, '-o', label='Sin')
gaborl, = plt.plot(x,gabor_err, '-^', label='Gabor')
plt.xlabel('迭代次数'); plt.ylabel('重构误差')
plt.legend(handles=[sinl, gaborl])
plt.show()


'''
##plot
mpl.rc('font', family='SimHei')
fig2, ((ax4, ax5))=  plt.subplots(nrows=1, ncols=2, sharex=False, figsize=(8,8))
ax4.set(title='sindb', xlabel='迭代次数', ylabel='重构误差')
ax4.plot(sindb_err, '-o')
ax5.set(title='gabor',xlabel='迭代次数', ylabel='重构误差')
ax5.plot(gabor_err, '-^')
# plt.legend(handles=[sin_line, gabor_line])
plt.show()
'''

'''
# save data to file
fname1 = 'output%ssignal_sin%s.pkl'%(os.sep, str(n))
fname2 = 'output%ssignal_gabor%s.pkl'%(os.sep, str(n))
with open (fname1, 'wb') as f:
    pickle.dump((y_pos, coefs, re_y, recon1), f)

with open(fname2, 'wb') as f:
    pickle.dump((y_pos2, coefs2, re_y2, recon2), f)
'''


