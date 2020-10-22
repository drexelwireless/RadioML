import pickle
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
from pyts.image import GADF, GASF
import tqdm
import os
import time
import gc
from numba import jit, cuda 

Xd = pickle.load(open("RML2016.10a_dict.pkl",'rb'), encoding='latin1')
# print(len(Xd[list(Xd.keys())[0]]))
snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
X = []  
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod,snr)])
        for i in range(Xd[(mod,snr)].shape[0]):  lbl.append((mod,snr))
X = np.vstack(X)
#we have labels for each one of them woowoo
#mods are all the classes are 11
#X.shape
plt.ioff()
path = os.path.join(os.getcwd())
signals = X

print(len(X))


# def main():
#     starttime = time.time()
#     pool = multiprocessing.Pool(15)
#     #pool.apply(dataset_creation, range(20))
#     for x in range(20):
#         pool.apply_async(dataset_creation, args=(x,))
#     #range(len(X)
#     pool.close()
#     pool.join()
#     pool.close()
#     print("="*35)
#     print('That took {} seconds'.format(time.time() - starttime))
# main()
#Trying to run on the GPU. Can 