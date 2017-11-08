import pandas as pd


class DataLoader(object):

    def retrieve_data_for_train(self, context, current_dt, data, start_dt):
        """
        Retrieve the data for the train, given a start date, the current datetime and the access to data

        :param TradingAlgorithm context:
        :param datetime.datetime current_dt:
        :param DataPortal data:
        :param datetime.datetime start_dt:
        :return:
        """

        historical_universes = None
        train_data = None

        return historical_universes, train_data

    def get_resampled_ohlcv_from_zipline(self, data, assets, bar_count, frequency, resample_rule, fill_limit,
                                         convert_symbols, dropna, fill_type='interpolate'):
        """
        Retrieve price history for a list of assets, fill gaps, resample, convert columns to symbols if required.
        :param data: BarData object containing current bar data for all assets in the universe.
        :param assets: list of assets to retrieve price history for.
        :param bar_count: number of bars to retrieve.
        :param frequency: basic frequency of the bars: '1m' or '1d'
        :param resample_rule: string rule for pandas DataFrame resamble method
        :param fill_limit: forward and backward interpolate gaps in data for a maximum of fill_limit points
        :param convert_symbols: boolean determining if to convert the assets to sstring symbols or not
        :param dropna: if True drops columns containing any nan after gaps-filling
        :param fill_type: string specifying the type of nan filling [None, 'interpolate', 'fill']
        :return: resampled data_dict of data frames
        """

        predict_data = None

        return predict_data

    def get_data(self, data_source, start_dt, current_dt):

        data = None

        return data




