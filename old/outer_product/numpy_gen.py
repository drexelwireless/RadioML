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


def create_dirs(folder_name, training_ready=False, test = False):
    classes = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']
    path = os.path.join(os.getcwd(), folder_name)
    # path = os.path.join(root, folder_name)
    ## write a conditional on what happens if it exists    
    if (os.path.isdir(path)):
        pass
        # for (_, dirnames, _) in os.walk(path):
        #     if dirnames in classes:
        #         classes.remove(dirnames)
        # for i in range(len(classes)):
        #     os.makedirs('%s/%s'%(path ,classes[i]), exist_ok=True)
    
    else:
        os.mkdir(path)
        create_dirs(folder_name)
    
    if test is True:
        for snr in range(-20, 20, 2):
            os.mkdir(folder_name+"/%d"%(snr))
            for clss in classes:
                os.mkdir(folder_name+"/%d/%s"%(snr, clss))
        
    if training_ready is True:
        # print("We got here!")
        path_ = os.path.join(os.getcwd(), "training_zone/"+folder_name)
        # print(path_)
        os.makedirs(path_, exist_ok = True)
        create_dirs(path_+'/test', test=True)
        create_dirs(path_+'/train')

def run(j, folder_name):
    SNR = lbl[j][1]
    wave_type = lbl[j][0]
    I = X[j][0]
    Q = X[j][1]
    IMG_SIZE = 128

    rgbArray = np.zeros((IMG_SIZE, IMG_SIZE, 3), 'uint8')
    rgbArray[..., 0] = np.outer(I, I)
    rgbArray[..., 1] = np.outer(Q, Q)
    rgbArray[..., 2] = np.outer(I, Q)

    numbr = str(j)
    countr = int(numbr[len(numbr)-3:])
    ## slicing the last 200 images of a 1000 image batch for the test set.
    if countr>800:
        plt.imsave(os.getcwd()+"/training_zone/"+folder_name+'/test/'+"%d/%s/%s_%d.png"%(SNR, wave_type, wave_type, j), rgbArray)
    else:
        plt.imsave(os.getcwd()+"/training_zone/"+folder_name+'/train/'+"%s/%s_%d.png"%(wave_type, wave_type, j), rgbArray)

def main():
    folder_name = 'dataset'
    create_dirs(folder_name, training_ready=True)
    pool = multiprocessing.Pool(12)
    runner = partial(run, folder_name = folder_name)
    count = 0
    for X in range(1000, 221000, 1000):
        # runner()
        count+=1
        # print(count)
        for _ in tqdm.tqdm(pool.imap_unordered(runner, range(X-1000, X)), total=1000):
            pass
    pool.close()
    pool.join()
    pool.close()
    
main()

