
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

Xd = pickle.load(open("../2016.04C.multisnr.pkl",'rb'), encoding='latin1')
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
    return encoded.reshape(1, 11)

train_img = []
test_img = []
test_lbl = []
test_snr = []
train_lbl = []

def run(j, folder_name):
    SNR = lbl[j][1]
    wave_type = lbl[j][0]
    I_ = X[j][0]
    Q_ = X[j][1]
    IMG_SIZE = 128
    array = NormalizeData(X[j])
    #I = NormalizeData(I)
    #Q = NormalizeData(Q)
    encoder1 = MTF(128, n_bins=IMG_SIZE//20, quantiles='gaussian')

    rgbArray = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), np.float32)
    
    a = np.squeeze(encoder1.fit_transform(X[j]))
    r = (a.reshape(-1, 1)).reshape((2, 128, 128))[0, :] #I
    
    rgbArray[..., 0] = NormalizeData(r)
    rgbArray[..., 1] = NormalizeData(np.outer(Q_, Q_))
    rgbArray[..., 2] = NormalizeData(np.outer(I_, Q_))

    numbr = str(j)
    countr = int(numbr[len(numbr)-3:])  
    if countr>580: #800 for a
        test_img.append(rgbArray)
        test_snr.append(SNR)
        test_lbl.append(one_hot_encoding(wave_type))
    else:
        train_img.append(rgbArray)
        train_lbl.append(one_hot_encoding(wave_type))

def main():
    folder_name = 'dataset'

    #create_dirs(folder_name, training_ready=True)
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    runner = partial(run, folder_name = folder_name)
    count = 0
    for X in range(730, len(signals)+730, 730):
	# for C dataset
	# for a dataset -> 1000, 221000, 1000
        # runner()
        count+=1
        # print(count)
        for i in tqdm.tqdm(range(X-730, X)):
            runner(i)

#        for _ in tqdm.tqdm(pool.imap_unordered(runner, range(X-730, X)), total=730):
#            pass
    path = "datasets/mtf_out_out" 
    os.makedirs(path + "/train")
    os.makedirs(path + "/test")
    np.save(path + "/test/test_img.npy", test_img)
    np.save(path + "/test/test_snr.npy", test_snr)
    np.save(path + "/test/test_lbl.npy", test_lbl)
    np.save(path + "/train/train_img.npy", train_img)
    np.save(path + "/train/train_lbl.npy", train_lbl)

    pool.close()
    pool.join()
    pool.close()

main()
