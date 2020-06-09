import pickle
import numpy as np
import multiprocessing
import numpy as np
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
warnings.simplefilter(action='ignore', category=FutureWarning)

Xd = pickle.load(open("RML2016.10a_dict.pkl",'rb'), encoding='latin1')
print(len(Xd[list(Xd.keys())[0]]))
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

def run(j):
    # for j in tqdm.tqdm(range(10)):
    wave_type = lbl[j][0]
    array = X[j]
    IMG_SIZE = 128
    uvts = PAA(output_size=IMG_SIZE).fit_transform(array)
    # encoder1 = GASF(IMG_SIZE)
    encoder1= RecurrencePlots()
    encoder2 = MTF(IMG_SIZE, n_bins=IMG_SIZE//20, quantiles='gaussian')
    encoder3 = GADF(IMG_SIZE)
        
    r = np.squeeze(encoder1.fit_transform(array)) 
    g = np.squeeze(encoder2.fit_transform(uvts))
    b = np.squeeze(encoder3.fit_transform(array))
    
    assert(r.shape==b.shape==g.shape)
    scaler = MinMaxScaler(feature_range=(0, 1))

    shape = r.shape
    r = scaler.fit_transform(r.reshape(-1, 1)).reshape(shape)[0, :]
    g = scaler.fit_transform(g.reshape(-1, 1)).reshape(shape)[0, :]
    b = scaler.fit_transform(b.reshape(-1, 1)).reshape(shape)[0, :]
    rgbArray = np.zeros((IMG_SIZE, IMG_SIZE, 3), 'uint8')
    rgbArray[..., 0] = r * 256
    rgbArray[..., 1] = g * 256
    rgbArray[..., 2] = b * 256
    plt.imsave(os.getcwd()+'/Datasets_Prod/3Encodings_GAFMTF0/'+"%s/%s_%d.png"%(wave_type, wave_type, j),rgbArray)
# run(X, lbl)

def main():
    pool = multiprocessing.Pool(12)
    training = range(len(X))
    for _ in tqdm.tqdm(pool.imap_unordered(run, training), total=len(training)):
        pass
    #range(len(X)
    pool.close()
    pool.join()
    pool.close()

main()