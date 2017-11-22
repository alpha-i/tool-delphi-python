import pandas_market_calendars as mcal

from alphai_finance.data.cleaning import convert_to_utc, select_trading_hours
from alphai_finance.data.read_from_hdf5 import get_all_table_names_in_hdf5, read_feature_data_dict_from_hdf5
from delphi.data_source.abstract_data_source import AbstractDataSource


class HDF5DataSource(AbstractDataSource):

    def __init__(self, config):
        super().__init__(config)

    def start(self):
        return self.config["start"]

    def end(self):
        return self.config["end"]

    def get_data(self, current_datetime, interval):

        nyse_market_calendar = mcal.get_calendar(self.config["exchange"])
        symbols = get_all_table_names_in_hdf5(self.config["filename"])

        start_datetime = current_datetime - interval
        end_datetime = current_datetime

        data_dict = read_feature_data_dict_from_hdf5(symbols, start_datetime, end_datetime, self.config["filename"])
        data_dict = convert_to_utc(data_dict)
        data_dict = select_trading_hours(data_dict, nyse_market_calendar)

        return data_dict
