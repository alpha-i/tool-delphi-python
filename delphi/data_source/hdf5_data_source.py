import datetime

import pandas_market_calendars as mcal
import pytz
from alphai_finance.data.cleaning import convert_to_utc, select_trading_hours
from alphai_finance.data.read_from_hdf5 import get_all_table_names_in_hdf5, read_feature_data_dict_from_hdf5

from delphi.data_source.abstract_data_source import AbstractDataSource


class StocksHDF5DataSource(AbstractDataSource):
    def __init__(self, configuration):
        super().__init__(configuration)
        self.timezone = pytz.utc
        self.filename = self.config["filename"]
        self.symbols = get_all_table_names_in_hdf5(self.filename)
        self.calendar = mcal.get_calendar(self.config["exchange"])

    def start(self):
        return self.config["start"]

    def end(self):
        return self.config["end"]

    def get_data(self, current_datetime, interval):

        assert current_datetime.tzinfo == self.timezone, "Datetime must provided in {} timezone".format(self.timezone)
        start_datetime = current_datetime - interval
        end_datetime = current_datetime

        data_dict = self._extract_data(self.symbols, start_datetime, end_datetime)

        return data_dict

    def _extract_data(self, symbols, start_datetime, end_datetime):
        data_dict = read_feature_data_dict_from_hdf5(symbols, start_datetime, end_datetime, self.filename,
                                                     self.timezone)
        data_dict = convert_to_utc(data_dict)
        return select_trading_hours(data_dict, self.calendar)

    def values_for_symbols_feature_and_time(self, symbol_list, feature, current_datetime):

        assert current_datetime.tzinfo == self.timezone, "Datetime must provided in {} timezone".format(self.timezone)

        data = self._extract_data(symbol_list, current_datetime - datetime.timedelta(minutes=1), current_datetime)
        try:
            values_for_features = data[feature]
        except KeyError as e:
            raise KeyError("Feature {} not present in data".format(feature))

        return values_for_features.loc[current_datetime]
