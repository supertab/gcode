import sys
sys.path.append('../scripts')
import mp_dicts
import todb
import mp2
import quota
import numpy as np
import os
import pickle
import time
import glob
import sys
import os.path as pth
import scipy.cluster.vq as vq
from PIL import Image
import matplotlib.pyplot as plt

sql = '''create table if not exists test(
    name char(20),
    k int,
    ssim float,
    filesize int,
    time float,
    primary key (name, k));'''
todb.create_table(sql)
name, k, ssim, size, time = 'lena', 5, 0.85, 128, 111.22
todb.insert(name, k, ssim, size, time, tbname='test')
