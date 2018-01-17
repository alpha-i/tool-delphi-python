import pandas as pd
import xarray as xr


def read_symbol_data_frame_from_hdf5(symbol, store):
    data_frame = store.select(symbol)[['close', 'volume', 'open', 'high', 'low']]
    data_frame = data_frame.reset_index().drop_duplicates(subset='datetime', keep='last').set_index('datetime')
    return data_frame


hdf5_file_name = "../tests/resources/19990101_19990301_3_stocks.hdf5"
hdf5_store = pd.HDFStore(hdf5_file_name)

symbols = [table[1:] for table in hdf5_store.keys()]

symbol_data_dict = {}
try:
    for symbol in symbols:
        symbol_data_dict[symbol] = read_symbol_data_frame_from_hdf5(symbol, hdf5_store)
except Exception as e:
    hdf5_store.close()
    raise e
hdf5_store.close()

panel = pd.Panel(symbol_data_dict)

panel.minor_axis.name = "raw_feature"

xray = xr.Dataset(panel)

xray_file_name = "../tests/resources/19990101_19990301_3_stocks.nc"
xray.to_netcdf(xray_file_name)