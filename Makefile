.PHONY : all
all : data/2016.10a.h5 data/2016.10b.h5 data/2016.04c.h5

CONVERT = python3 bin/convert-radioml2016.py

data/2016.10a.h5 : data/deepsig/RML2016.10a_dict.pkl
	$(CONVERT) $< $@ -d

data/2016.10b.h5 : data/deepsig/RML2016.10b.dat
	$(CONVERT) $< $@ -d

data/2016.04c.h5 : data/deepsig/2016.04C.multisnr.pkl
	$(CONVERT) $< $@ -d
