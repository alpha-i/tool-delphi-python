import pandas_market_calendars as mcal

from alphai_finance.data.cleaning import convert_to_utc, select_trading_hours
from alphai_finance.data.read_from_hdf5 import get_all_table_names_in_hdf5, read_feature_data_dict_from_hdf5

from alphai_prototype_env.data_providers.base import AbstractDataProvider


class HDF5DataProvider(AbstractDataProvider):

    def __init__(self, data_file, exchange):
        self._data_file = data_file
        self._exchange = exchange

    def get_data(self, start_date, end_date):

        nyse_market_calendar = mcal.get_calendar(self._exchange)
        symbols = get_all_table_names_in_hdf5(self._data_file)

        data_dict = read_feature_data_dict_from_hdf5(symbols, start_date, end_date, self._data_file)
        data_dict = convert_to_utc(data_dict)
        data_dict = select_trading_hours(data_dict, nyse_market_calendar)

        return data_dict
