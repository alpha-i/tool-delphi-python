"""
Scrip to convert easily a HDF5 file (in our usual format) to a XArray source file

Usage: python convert_hdf5_to_xarray_netcdf.py <input path.hdf5> <output_path.nc>
"""

import argparse
import logging
import os

import pandas as pd
import xarray as xr

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description='Process a HDF5 file and convert it to a xarray')

parser.add_argument('input', metavar='input', type=str, help='the input file path')
parser.add_argument('output', metavar='output', type=str, help='The destination file path')


def read_symbol_data_frame_from_hdf5(symbol, store):
    data_frame = store.select(symbol)[['close', 'volume', 'open', 'high', 'low']]
    data_frame = data_frame.reset_index().drop_duplicates(subset='datetime', keep='last').set_index('datetime')
    return data_frame


if __name__ == '__main__':
    args = parser.parse_args()
    HDF5_FILE_NAME = args.input
    XRAY_FILE_NAME = args.output

    if not os.path.exists(HDF5_FILE_NAME):
        raise Exception("No such file :%s", HDF5_FILE_NAME)

    logging.info("Opening input file: %s", HDF5_FILE_NAME)
    hdf5_store = pd.HDFStore(HDF5_FILE_NAME)

    symbols = [
        table[1:] for table in hdf5_store.keys()
    ]

    symbol_data_dict = {}
    try:
        for symbol in symbols:
            symbol_data_dict[symbol] = read_symbol_data_frame_from_hdf5(symbol, hdf5_store)
    finally:
        hdf5_store.close()

    panel = pd.Panel(symbol_data_dict)
    panel.minor_axis.name = "raw_feature"
    xray = xr.Dataset(panel)

    logging.info("Writing output file: %s", XRAY_FILE_NAME)
    xray.to_netcdf(XRAY_FILE_NAME)
    xray.close()
    logging.info("File converted successfully!")
