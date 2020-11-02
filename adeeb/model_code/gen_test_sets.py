import numpy as np
import config
import os

test_img = np.load("../test_img.npy")
test_lbl = np.load("../test_lbl.npy")
test_snr = np.load("../test_snr.npy")

for i in range(-20, 20, 2):
	path = os.getcwd() + '/test/' + str(i) 
	os.makedirs(path)
	img = []
	lbl = []
	for j in range(len(test_snr)):
		if i==test_snr[j]:
			img.append(test_img[j])
			lbl.append(test_lbl[j])
	np.save(path+"/test_img_"+ str(i)+".npy", img)
	np.save(path+"/lbl_img_"+ str(i)+".npy", lbl)    

