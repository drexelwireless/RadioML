import tensorflow as tf 
import os
import csv
import numpy as np


models = ["model/mtf_out_out.h5","model/gadf_out_out.h5", "model/gasf_out_out.h5", "model/out_out_out.h5" ]

for mdl in [models[1]]:
    ##loading the model
    model = tf.keras.models.load_model(mdl)
    encoding_scheme = mdl.split("/")[1].split(".")[0]
    with open(encoding_scheme+".csv", "w") as wr:
        wr = csv.writer(wr)
        for path, i, files in os.walk(f"datasets/{encoding_scheme}"):
            if "test" in path:
                if len(files)>0:
                    files = sorted(files)
                    lbl = np.load(path + "/" + files[0])
                    img = np.load(path + "/" + files[1])
                    snr = path.split("/")[3]
                    ## calculating the results for each snr:
                    data = tf.data.Dataset.from_tensor_slices((img, lbl))
                    scores = model.evaluate(data, verbose=1)
                    results_list = [snr, scores[1]]
                    print(f"{encoding_scheme} == {results_list}")
                    wr.writerow(results_list)
