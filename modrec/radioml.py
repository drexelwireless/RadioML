import h5py
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

def to_hdf(data : Dict[Tuple[str, float], np.ndarray], path : Path, name : str = 'radioml'):
    """Save Python dictionary containing RadioML data to HDF5.

    Save RadioML data represented as a dictionary mapping (modulation, snr) keys
    to IQ data as an HDF5 file.

    Each record in the HDF5 dataset will have three attributes: 'ms', the
    modulation scheme, represented as an integer category index, 'snr', the SNR,
    and 'iq_data', the complex IQ data. The modulation categories are stored as
    a list of strings in the 'ms_categories' attribute of the dataset.

    Args:
        data: RadioML data dictionary
        path: Path to HDF5 file
        name: Name of dataset
    """
    # Determine modulation categories
    categories = sorted(set(ms for ms, _ in data.keys()))
    logger.info("Modulation schemes: %s", categories)

    # Stack all data
    iq_data_items = []
    ms_items = []
    snr_items = []

    for ms, snr in data.keys():
        iq_data = data[(ms,snr)]
        iq_data_items.append(iq_data)
        ms_items.append(np.full((iq_data.shape[0]), categories.index(ms)))
        snr_items.append(np.full((iq_data.shape[0]), snr))

        logger.debug("Modulation = %s, SNR = %g, samples = %d", ms, snr, len(iq_data))

    iq_data = np.vstack(iq_data_items)
    ms = np.concatenate(ms_items)
    snr = np.concatenate(snr_items)

    # Create structured numpy array for data
    dt = np.dtype([ ('ms', ms.dtype)
                  , ('snr', snr.dtype)
                  , ('iq_data', iq_data.dtype, iq_data.shape[1:])
                  ])

    arr = np.empty(ms.shape[0], dt)
    arr['ms'] = ms
    arr['snr'] = snr
    arr['iq_data'] = iq_data

    save_numpy(categories, arr, path, name)

def save_numpy(categories : List[str], arr : np.ndarray, path : Path, name : str = 'radioml'):
    """Save RadioML numpy data as HDF5"""
    # Save to HDF5
    with h5py.File(path, 'w') as f:
        logger.info("Saving to HDF5: %s", name)

        dset = f.create_dataset(name, (len(arr),), arr.dtype)
        dset.attrs['ms_categories'] = categories
        dset.write_direct(arr)

def load_numpy(path : Path, name : str = 'radioml') -> Tuple[List[str], np.ndarray]:
    """Read RadioML data stored as HDF5 as a structured numpy array.

    Args:
        path: Path to HDF5 file
        name: Name of dataset

    Returns:
        A pair of list of categories and a structured numpy array containing
        RadioML data.
    """
    with h5py.File(path, 'r') as f:
        ds = f[name]
        ms_categories = ds.attrs['ms_categories']

        data = np.empty(len(ds), dtype=ds.dtype)
        ds.read_direct(data)

    return (ms_categories, data)

def load_dataframe(path : Path, name : str = 'radioml') -> pd.DataFrame:
    """Read RadioML data stored as HDF5 as a pandas DataFrame.

    Args:
        path: Path to HDF5 file
        name: Name of dataset

    Returns:
        A Pandas DataFrame containing RadioML data.
    """
    ms_categories, data = load_numpy(path, name)

    df = pd.DataFrame()

    df['ms'] = data['ms']
    df['snr'] = data['snr']
    df['iq_data'] = list(interleave_float32(data['iq_data']))

    ms_cats = pd.CategoricalDtype(range(0, len(ms_categories)))
    df.ms = df.ms.astype(ms_cats)
    df.ms.cat.rename_categories(ms_categories, inplace=True)

    return df

def interleave_float32(x):
    """Interleave float32 array of shape (2, N) to produce a complex64 array of shape (N,)"""
    return np.ascontiguousarray(x.transpose(0,2,1)).view('complex64').squeeze()

def uninterleave_complex64(x):
    """Un-interleave complex64 data, converting a complex64 array of shape (N,) to a float32 array of shape (2, N)"""
    return np.ascontiguousarray(x.view('float32').reshape(128,2).transpose(1,0))
