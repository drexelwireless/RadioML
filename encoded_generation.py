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
import shutil
from functools import partial

warnings.simplefilter(action='ignore', category=FutureWarning)

Xd = pickle.load(open("RML2016.10a_dict.pkl",'rb'), encoding='latin1')
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

def create_dirs(folder_name, training_ready=False):
    classes = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']
    root = os.path.join(os.getcwd(), 'Datasets_Prod')
    path = os.path.join(root, folder_name)
    ## write a conditional on what happens if it exists
    if (os.path.isdir(path)):
        for (_, dirnames, _) in os.walk(path):
            if dirnames in classes:
                classes.remove(dirnames)
        for i in range(len(classes)):
            os.makedirs('%s/%s'%(path ,classes[i]), exist_ok=True)
    else:
        os.mkdir(path)
        create_dirs(folder_name)
    if training_ready is True:
        # print("We got here!")
        path_ = os.path.join(os.getcwd(), "training_zone/"+folder_name)
        # print(path_)
        os.makedirs(path_, exist_ok = True)
        create_dirs(path_+'/test')
        create_dirs(path_+'/train')
        
def run(j, folder_name):
    # for j in tqdm.tqdm(range(10)):
    # if lbl[j][1] == -6:
    wave_type = lbl[j][0]
    array = X[j]
    IMG_SIZE = 128
    uvts = PAA(output_size=IMG_SIZE).fit_transform(array)
    encoder1 = GASF(IMG_SIZE)
    # encoder1= RecurrencePlots()
    encoder2 = MTF(IMG_SIZE, n_bins=IMG_SIZE//20, quantiles='gaussian')
    # encoder2 = RecurrencePlots()
    encoder3 = GADF(IMG_SIZE)
        
    r = np.squeeze(encoder1.fit_transform(array)) 
    g = np.squeeze(encoder2.fit_transform(uvts)) #uvts->array
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
    plt.imsave(os.getcwd()+'/Datasets_Prod/'+folder_name+'/'+"%s/%s_%d.png"%(wave_type, wave_type, j),rgbArray)


def moving(folder_name):
    path = os.path.join(os.getcwd(), folder_name)
    path_test =  os.getcwd()+'/%s/%s'%('training_zone', folder_name)+'/test'
    path_train = os.getcwd()+'/%s/%s'%('training_zone', folder_name)+'/train'
    # num_files = len(next(os.walk(path+"/training_zone"))) # number of files in the directory
    classes = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']
    for i in os.walk(path):
        if folder_name in i[0]:
            name = i[0]
            # print(name)
            abcd = name.split('/')
            mod_name = abcd[len(abcd) - 1]
            if mod_name in classes:
                count = 0
                for img in i[2]:
                    if count < 16000:
                        shutil.copy(name+ "/" +img, path_train+'/'+mod_name)
                    else:
                        shutil.copy(name+ "/" +img, path_test+'/'+mod_name)
                    count+=1
                print(mod_name, "images moved", count)

def main():
    folder_name = 'new_hampshire'
    create_dirs(folder_name, training_ready=True)
    pool = multiprocessing.Pool(12)
    training = range(len(X))
    runner = partial(run, folder_name = folder_name)
    for _ in tqdm.tqdm(pool.imap_unordered(runner, training), total=len(training)):
        pass
    #range(len(X)
    pool.close()
    pool.join()
    pool.close()
    # moving(folder_name)
main()