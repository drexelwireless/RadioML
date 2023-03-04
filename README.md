# `modrec` [![Python package](https://github.com/drexelwireless/RadioML/actions/workflows/python-package.yml/badge.svg)](https://github.com/drexelwireless/RadioML/actions/workflows/python-package.yml)

Run the following commands after downloading the datasets from deepsig.io- 

```
pip install -e .
make
```

For training - 
![image](https://user-images.githubusercontent.com/38449494/125520596-732cfffb-817c-435b-9971-c6102886fe8a.png)

Example -- 
```
python3 bin/train.py data/2016.10a.h5 --train --model resnet18-outer --batch-size 512 
```

### To Cite 
```@INPROCEEDINGS{10017640,
  author={Abbas, Adeeb and Pano, Vasil and Mainland, Geoffrey and Dandekar, Kapil},
  booktitle={MILCOM 2022 - 2022 IEEE Military Communications Conference (MILCOM)}, 
  title={Radio Modulation Classification Using Deep Residual Neural Networks}, 
  year={2022},
  volume={},
  number={},
  pages={311-317},
  doi={10.1109/MILCOM55135.2022.10017640}}
```
This repo also has other supporting tools like data loaders/saving it in hdf5 etc that were used during the work - https://ieeexplore.ieee.org/abstract/document/10017640/

There are no active maintainers of this project. For any queries/concerns, email - adeeb@drexel.edu
