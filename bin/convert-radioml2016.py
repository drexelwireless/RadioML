#!/bin/env python3
import argparse
import logging
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf

import modrec.radioml

def main():
    parser = argparse.ArgumentParser(description='Convert RadioML 2016 data to standard HDF5 format.')
    parser.add_argument('-d', '--debug', action='store_const', const=logging.DEBUG,
                        dest='loglevel',
                        default=logging.WARNING,
                        help='print debugging information')
    parser.add_argument('-v', '--verbose', action='store_const', const=logging.INFO,
                        dest='loglevel',
                        help='be verbose')
    parser.add_argument('input', help='RadioML pickle-format data')
    parser.add_argument('output', help='HDF5 output')

    # Parse arguments
    try:
        args = parser.parse_args()
    except SystemExit as ex:
        return ex.code

    # Set up logging
    logging.basicConfig(format='%(asctime)s:%(name)s:%(levelname)s:%(message)s',
                        level=args.loglevel)

    # Load RadioML 2016 data setfile
    data = pickle.load(open(args.input, 'rb'), encoding='latin1')

    modrec.radioml.to_hdf(data, args.output)

if __name__ == "__main__":
    main()
