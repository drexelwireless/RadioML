# RadioML

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
