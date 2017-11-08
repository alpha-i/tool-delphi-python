import pandas_market_calendars as mcal

from alphai_finance.data.read_from_hdf5 import(
    read_symbol_data_dict_from_hdf5,
    read_feature_data_dict_from_hdf5,
    get_all_table_names_in_hdf5
)

from alphai_finance.data.cleaning import (
    select_trading_hours,
    convert_to_utc
)


class StockLoader(object):

    def __init__(self):
        self.name = "stock_loader"
        self.train_start = '19990101'
        self.train_end = '19991001'
        self.test_start = '19991002'
        self.test_end = '20000101'
        self.filename = '/home/rmason/Downloads/19990101_2000_0101_100_stocks.hdf5'
        self.nyse_market_calendar = mcal.get_calendar('NYSE')
        self.symbols = get_all_table_names_in_hdf5(self.filename)
        self.train_data = self.get_filtered_data(self.train_start, self.train_end)
        self.test_data = self.get_filtered_data(self.test_start, self.test_end)

    def get_filtered_data(self, t_start, t_end):
        """
        :param t_start: string
        :param t_end: string
        :param resample_rate: string
        :return: dict of dataframes with keys 'open', 'high',...,'volume'
        """
        data = read_feature_data_dict_from_hdf5(self.symbols, t_start, t_end, self.filename)
        data = convert_to_utc(data)
        data = select_trading_hours(data, self.nyse_market_calendar)

        return data

if __name__ == '__main__':

    stock_loader = StockLoader()

