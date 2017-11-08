import pandas_market_calendars as mcal

from alphai_finance.data.read_from_hdf5 import(
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
        self.train_data = self.get_train_data()
        self.test_data = self.get_test_data()

    def get_train_data(self):
        """
        :param t_start: string
        :param t_end: string
        :param resample_rate: string
        :return: dict of dataframes with keys 'open', 'high',...,'volume'
        """

        train_start = '19990101'
        train_end = '19991001'

        filename = '/home/rmason/Downloads/19990101_2000_0101_100_stocks.hdf5'
        nyse_market_calendar = mcal.get_calendar('NYSE')
        symbols = get_all_table_names_in_hdf5(filename)

        data = read_feature_data_dict_from_hdf5(symbols, train_start, train_end, filename)
        data = convert_to_utc(data)
        data = select_trading_hours(data, nyse_market_calendar)

        return data

    def get_test_data(self):
        """
        :param t_start: string
        :param t_end: string
        :param resample_rate: string
        :return: dict of dataframes with keys 'open', 'high',...,'volume'
        """

        test_start = '19991002'
        test_end = '20000101'

        filename = '/home/rmason/Downloads/19990101_2000_0101_100_stocks.hdf5'
        nyse_market_calendar = mcal.get_calendar('NYSE')
        symbols = get_all_table_names_in_hdf5(filename)

        data = read_feature_data_dict_from_hdf5(symbols, test_start, test_end, filename)
        data = convert_to_utc(data)
        data = select_trading_hours(data, nyse_market_calendar)

        return data




if __name__ == '__main__':

    stock_loader = StockLoader()

