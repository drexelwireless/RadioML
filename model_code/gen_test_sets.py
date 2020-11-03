import numpy as np
import config
import os
from tqdm import tqdm
from pathlib import Path

encoding_schemes = ['out_out_out', 'gasf_out_out', 'gadf_out_out', 'mtf_out_out']

for scheme in encoding_schemes:

    img = f"../datasets/{scheme}/test/test_img.npy"
    lbl = f"../datasets/{scheme}/test/test_lbl.npy"
    snr = f"../datasets/{scheme}/test/test_snr.npy"
    
    
    test_img = np.load(f"../datasets/{scheme}/test/test_img.npy")
    test_lbl = np.load(f"../datasets/{scheme}/test/test_lbl.npy")
    test_snr = np.load(f"../datasets/{scheme}/test/test_snr.npy")

    for i in tqdm(range(-20, 20, 2)):
            path = f"../datasets/{scheme}/test/{i}"
            os.makedirs(path, exist_ok=True)
            img = []
            lbl = []
            for j in range(len(test_snr)):
                if i==test_snr[j]:
                    img.append(test_img[j])
                    lbl.append(test_lbl[j])
            np.save(path+"/test_img_"+ str(i)+".npy", img)
            np.save(path+"/lbl_img_"+ str(i)+".npy", lbl)    
    #Path(str(img)).unlink()
    #Path(str(lbl)).unlink()
    #Path(str(snr)).unlink()
