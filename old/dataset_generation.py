import pickle
import numpy as np
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from pyts.image import GASF, GADF #.7 is when we 
import tqdm
import os
import time
import gc
import os
import csv

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
#we have labels for each one of them woowoo
#mods are all the classes are 11
#X.shape
len(list(lbl))
plt.ioff()
path = os.path.join(os.getcwd())
signals = X
# appropriate folder generation here
print(os.getcwd())
path = os.getcwd() + str('/dataset_128_gasf/')
os.mkdir(path)
for i in mods:
    # new_dirs = path + str(i) + "/GADF"
    os.makedirs(path+str(i))
    # new_dirs = path + str(i) + "/GASF"
#     os.makedirs(new_dirs)
# # 

def dataset_creation(j):
    wave_type = lbl[j][0]
    values = signals[j][:]
    image_size = 128
    
    gasf = GASF(image_size)
    X_gasf = gasf.fit_transform(values)
    # gadf = GADF(image_size)
    # X_gadf = gadf.fit_transform(values)

    wave_id = "%s_%d"%(wave_type, j) 

    plt.figure(figsize=(4, 4))
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.tight_layout()
    
    fig = plt.imshow(X_gasf[0], cmap='rainbow', origin='lower')
    fig.figure.savefig("%s/%s/%s.png"%(path, wave_type, wave_id), bbox_inches = 'tight', pad_inches = 0)
    # fig1 = plt.imshow(X_gadf[0], cmap='rainbow', origin='lower')
    # fig.figure.savefig("%s/dataset/%s/GADF/%s.png"%(path, wave_type, wave_id), bbox_inches = 'tight', pad_inches = 0)

        # Clear the current axes.
    plt.cla() 
        # Clear the current figure.
    plt.clf() 
        # Closes all the figure windows.
    plt.close('all')

def main():
    pool = multiprocessing.Pool(8)
    training = range(len(X))
    for _ in tqdm.tqdm(pool.imap_unordered(dataset_creation, training), total=len(training)):
        pass
    #range(len(X)
    pool.close()
    pool.join()
    pool.close()

    # with open("dataset_gasf.txt", "w") as outfile:
    #     writer = csv.writer(outfile, escapechar=' ', quoting=csv.QUOTE_NONE)
    #     writer.writerow(["image_name", "tags"])
    #     writer.writerow([gasf_list, labelz])

    # with open("dataset_gadf.txt", "w") as outfile:
    #     writer = csv.writer(outfile, escapechar=' ', quoting=csv.QUOTE_NONE)
    #     writer.writerow(["image_name", "tags"])
    #     writer.writerow([gadf_list, labelz])

    # with open('paths_gadf.txt', 'w') as filehandle:
    #     for listitem in paths_gadf:
    #         filehandle.write('%s\n' % listitem)
    
    # with open('paths_gasf.txt', 'w') as filehandle:
    #     for listitem in paths_gasf:
    #         filehandle.write('%s\n' % listitem)

main()
    
    