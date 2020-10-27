import pickle
import numpy as np
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import os
import gc
import os
import warnings
import shutil
from functools import partial
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from pyts.image import GADF, MTF, RecurrencePlots, GASF
from npy_append_array import NpyAppendArray


warnings.simplefilter(action='ignore', category=FutureWarning)

Xd = pickle.load(open("../RML2016.10a_dict.pkl",'rb'), encoding='latin1')
snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
X = []  
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod,snr)])
        for i in range(Xd[(mod,snr)].shape[0]):  lbl.append((mod,snr))
X = np.vstack(X)
plt.ioff()
path = os.path.join(os.getcwd())
signals = X

def NormalizeData(data):
    return (data + 1) / (2)

def one_hot_encoding(_lbls):
    encoded = np.zeros(11, dtype=int)
    classes = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']
    encoded[classes.index(_lbls)] = 1
    return encoded

train_img = []
test_img = []
test_lbl = []
test_snr = []
train_lbl = []

def run(j, folder_name):
    SNR = lbl[j][1]
    wave_type = lbl[j][0]
    I = X[j][0]
    Q = X[j][1]
    IMG_SIZE = 128

    array = NormalizeData(X[j])
    I = NormalizeData(I)
    Q = NormalizeData(Q)
    encoder1 = GASF(128)

    rgbArray = np.zeros((IMG_SIZE, IMG_SIZE, 3), np.float32)
    
    a = np.squeeze(encoder1.fit_transform(array))
    r = (a.reshape(-1, 1)).reshape((2, 128, 128))[0, :] #I
    r = NormalizeData(r)
    
    rgbArray[..., 0] = r
    rgbArray[..., 1] = np.outer(Q, Q)
    rgbArray[..., 2] = np.outer(I, Q)

    numbr = str(j)
    countr = int(numbr[len(numbr)-3:])
    if countr>800:
        test_img.append(rgbArray)
        test_snr.append(SNR)
        test_lbl.append(wave_type)
    else:
        train_img.append(rgbArray)
        train_lbl.append(wave_type)


def main():
    folder_name = 'dataset'


    #create_dirs(folder_name, training_ready=True)
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    runner = partial(run, folder_name = folder_name)
    count = 0
    for X in range(1000, 221000, 1000):
        # runner()
        count+=1
        # print(count)
        for i in tqdm.tqdm(range(X-1000, X)):
            runner(i)

#         for _ in tqdm.tqdm(pool.imap_unordered(runner, range(X-1000, X)), total=1000):
#             pass
    np.save("test_img.npy", test_img)
    np.save("test_snr.npy", test_snr)
    np.save("test_lbl.npy", test_lbl)
    np.save("train_img.npy", train_img)
    np.save("train_lbl.npy", train_lbl)

    pool.close()
    pool.join()
    pool.close()

main()
