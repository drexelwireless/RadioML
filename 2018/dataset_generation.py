import pickle
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
from pyts.approximation import PAA
from pyts.image import GADF, MTF, RecurrencePlots, GASF
from sklearn.preprocessing import MinMaxScaler
import tqdm
import os
import time
import gc
import os
import csv
import warnings
import shutil
from functools import partial
from numba import jit, cuda

warnings.simplefilter(action='ignore', category=FutureWarning)
tr_labels = np.load("./dataset_2018/test/labels.npy")
tr_signals = np.load("./dataset_2018/test/signals.npy")
length = len(tr_signals)
tr_snrs = np.load("./dataset_2018/test/snrs.npy")

classes = ['32PSK',
 '16APSK',
 '32QAM',
 'FM',
 'GMSK',
 '32APSK',
 'OQPSK', #1
 '8ASK',
 'BPSK',
 '8PSK',
 'AM-SSB-SC',
 '4ASK',
 '16PSK',
 '64APSK',
 '128QAM',
 '128APSK',
 'AM-DSB-SC', #2
 'AM-SSB-WC',
 '64QAM',
 'QPSK',
 '256QAM',
 'AM-DSB-WC',
 'OOK',
 '16QAM']

SNRs = [-20, -18, -16, -14, -12, -10,  -8,  -6,  -4,  -2,   0,   2,   4,
         6,   8,  10,  12,  14,  16,  18,  20,  22,  24,  26,  28,  30]

def create_dirs():
    for j in SNRs:
        for i in classes:
            os.makedirs("./img_data_testing/"+str(j)+'/'+i)

# create_dirs()
def run(i, folder_name, classes):
    db = str(tr_snrs[i])    
    wave_type = classes[np.where(tr_labels[i]==1)[0][0]]
    array = tr_signals[i].transpose()
    IMG_SIZE = 128
    encoder1 = GASF(IMG_SIZE)
    encoder2 = MTF(IMG_SIZE, n_bins=IMG_SIZE//20, quantiles='gaussian')
    encoder3 = GADF(IMG_SIZE)
    r = np.squeeze(encoder1.fit_transform(array)) 
    g = np.squeeze(encoder2.fit_transform(array))
    b = np.squeeze(encoder3.fit_transform(array))
    assert(r.shape==b.shape==g.shape)
    scaler = MinMaxScaler(feature_range=(0, 1))
    shape = r.shape
    r = scaler.fit_transform(r.reshape(-1, 1)).reshape(shape)[1, :]
    g = scaler.fit_transform(g.reshape(-1, 1)).reshape(shape)[0, :]
    b = scaler.fit_transform(b.reshape(-1, 1)).reshape(shape)[0, :]
    rgbArray = np.zeros((IMG_SIZE, IMG_SIZE, 3), 'uint8')
    rgbArray[..., 0] = r * 256
    rgbArray[..., 1] = g * 256
    rgbArray[..., 2] = b * 256
    plt.imsave(os.getcwd()+'/'+folder_name+'/'+"%s/%s/%s_%d.png"%(db, wave_type, wave_type, i),rgbArray)

def main():
    folder_name = "img_data_testing"
    pool = multiprocessing.Pool(os.cpu_count())
    training = range(length)
    runner = partial(run, folder_name = folder_name, classes = classes)
    for _ in tqdm.tqdm(pool.imap_unordered(runner, training), total=len(training)):
        pass
    pool.close()
    pool.join()
    pool.close()
main()
