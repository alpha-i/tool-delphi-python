from copy import deepcopy
from datetime import timedelta

import pytz

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

from alphai_delphi import AbstractDataSource


def _make_zero_series(x_array):
    """
    :param x_array: y
    :return: ndarray of zeroes with length
    """
    return np.zeros(len(x_array))


def _make_constant_series(x_array, constant=0.5):
    """

    :param constant: value of constant series
    :param x_array: length of the array to be made
    :return: ndarray of constant values
    """

    return _make_zero_series(x_array=x_array) + constant


def _make_linear_series(x_array, slope=1., intercept=0.):
    """

    :param x_array: x_values
    :param slope:  slope of the line
    :param intercept: intercept of the line
    :return: ndarray of linear
    """
    return (x_array * slope) + intercept


def _make_sin_series(x_array):
    """

    :param x_array: x_values
    :return: ndarray of sin values
    """
    return np.sin(x_array)


def _make_cos_series(x_array):
    """

    :param x_array: x_values
    :return: ndarray of cos values
    """
    return np.cos(x_array)


def create_minute_datetime_index(exchange_name, start_date, end_date):
    calendar = mcal.get_calendar(exchange_name)
    schedule = calendar.schedule(start_date, end_date)

    datetime_index = pd.DatetimeIndex([])

    for idx in range(len(schedule)):
        start_minute = schedule.market_open[idx] + timedelta(minutes=1)
        end_minute = schedule.market_close[idx]
        datetime_index = datetime_index.append(pd.date_range(start=start_minute, end=end_minute, freq='min'))

    return datetime_index


def _create_sin_dict(x_array, n_series):
    """

    :param x_array:
    :param n_series:
    :return:
    """

    sin_dict = {}
    for i in range(n_series):
        name = 'sin_' + str(i)
        param1 = 2.10 - (0.05 * i)
        param2 = 0.15 + (0.05 * i)
        param3 = 0.00005 + (0.000001 * i)
        data = -0.000001 + _make_sin_series(x_array * param1 + param2) * param3
        sin_dict[name] = data

    return sin_dict


def _make_distinct_series(start_date, end_date, n_series, add_zero_and_linear=False):
    """

    :param start_date: start date as string
    :param end_date: end date as a string
    :param n_series: how many sin series do you want
    :return: dataframe with 4 distinct time series data for week days only
    """
    time_index = create_minute_datetime_index('NYSE', start_date, end_date)
    x_array = np.linspace(0, 100, num=len(time_index))

    sin_dict = _create_sin_dict(x_array, n_series)

    if add_zero_and_linear:

        data_dict = {
            'zero': _make_zero_series(x_array),
            'linear1': _make_linear_series(x_array, slope=0.00000005, intercept=0.000001),
            'linear2': _make_linear_series(x_array, slope=-0.0000001, intercept=-0.000001),
        }
        data_dict.update(sin_dict)
    else:
        data_dict = sin_dict

    data_frame = pd.DataFrame(data=data_dict, index=time_index)

    return data_frame


def convert_log_returns_to_prices(log_returns):
    """
    based on the dataframe with log returns calculate the prices that will produce the given log-returns

    :param log_returns: dataframe with log_returns
    :return: dataframe with prices that will have the provided log_returns
    """
    initial_price = 1
    return initial_price * np.exp(log_returns.cumsum())


def make_ohlcv_dict(dataframe):
    """

    :param dataframe: pandas dataframe
    :return: dict that with keys open high low close volume and the given dataframe as value
    """
    open = deepcopy(dataframe)

    high = deepcopy(dataframe)
    high = 1.1 * high

    low = deepcopy(dataframe)
    low = 0.9 * low

    close = deepcopy(dataframe)
    close.values[:-1, :] = open.values[1:, :]

    volume = deepcopy(dataframe)
    volume = volume.round(0).astype(int) + 1

    ohlcv_dict = {'open': open, 'high': high, 'low': low, 'close': close, 'volume': volume}

    return ohlcv_dict


def add_nans(ohlcv_dict):
    """

    :param ohlcv_dict:
    :return:
    """
    # TODO make this code nicer
    for key, value in ohlcv_dict.items():

        value['ALL_NAN'] = value['sin_8'].astype(float)
        value.loc[:, 'ALL_NAN'] = np.NAN
        value.drop('sin_8', axis=1, inplace=True)

        value['ONE_NAN'] = value['sin_9'].astype(float)
        value.loc[:, 'ONE_NAN'][33060] = np.NAN
        value.drop('sin_8', axis=1, inplace=True)

        value['10_NAN'] = value['sin_10'].astype(float)
        value.loc[:, '10_NAN'][33060:33060 + 10] = np.NAN
        value.drop('sin_10', axis=1, inplace=True)

        value['20_NAN'] = value['sin_11'].astype(float)
        value.loc[:, '20_NAN'][33060: 33060 + 20] = np.NAN
        value.drop('sin_11', axis=1, inplace=True)

        value['30_NAN'] = value['sin_12'].astype(float)
        value.loc[:, '30_NAN'][33060: 33060 + 30] = np.NAN
        value.drop('sin_12', axis=1, inplace=True)

        value['400_NAN'] = value['sin_13'].astype(float)
        value.loc[:, '400_NAN'][33060: 33060 + 400] = np.NAN
        value.drop('sin_13', axis=1, inplace=True)

        value['TAIL_NAN'] = value['sin_15'].astype(float)
        value.loc[:, 'TAIL_NAN'][80000:] = np.NAN
        value.drop('sin_15', axis=1, inplace=True)

    return ohlcv_dict


def create_synthetic_ohlcv(n_sin_series, start_date='20100101', end_date='20100131', add_nan=False,
                           add_zero_and_linear=False):
    """

    :param add_nan:
    :param n_sin_series:
    :param add_zero_and_linear:
    :param start_date:
    :param end_date:
    :return:
    """
    log_returns = _make_distinct_series(start_date, end_date, n_series=n_sin_series,
                                        add_zero_and_linear=add_zero_and_linear)
    prices = convert_log_returns_to_prices(log_returns)
    ohlcv_dict = make_ohlcv_dict(prices)
    if add_nan:
        assert n_sin_series >=16, 'If adding nan n_sin_series must be >= 16'
        ohlcv_dict = add_nans(ohlcv_dict)

    return ohlcv_dict


class SyntheticDataSource(AbstractDataSource):

    @property
    def start(self):
        return self.start_date

    @property
    def end(self):
        return self.end_date

    DEFAULT_ADD_NAN = False
    DEFAULT_N_SIN_SERIES = 3
    DEFAULT_ADD_ZERO_AND_LINEAR = False

    def __init__(self, configuration):
        """
        {}
        start_date
        end_date
        add_nan,
        n_sin_series=3,
        add_zero_and_linear=False

        """
        super().__init__(configuration)

        self.start_date = configuration['start_date']
        self.end_date = configuration['end_date']
        self.add_nan = configuration.get('add_nan', self.DEFAULT_ADD_NAN)
        self.n_sin_series = configuration.get('n_sin_series', self.DEFAULT_N_SIN_SERIES)
        self.add_zero_and_linear = configuration.get('add_zero_and_linear', self.DEFAULT_ADD_ZERO_AND_LINEAR)
        self._data_dict = create_synthetic_ohlcv(
            start_date=self.start_date,
            end_date=self.end_date,
            add_nan=self.add_nan,
            n_sin_series=self.n_sin_series,
            add_zero_and_linear=self.add_zero_and_linear

        )

    def get_data(self, current_datetime, interval):
        assert current_datetime.tzinfo == pytz.utc, "Datetime must provided in UTC timezone"
        start_datetime = current_datetime - interval
        end_datetime = current_datetime

        data = {}
        for key in self._data_dict.keys():
            data[key] = self._data_dict[key][start_datetime:end_datetime]
        return data

    def values_for_symbols_feature_and_time(self, symbol_list, feature, current_datetime):
        assert current_datetime.tzinfo == pytz.utc, "Datetime must provided in UTC timezone"

        try:
            values_for_features = self._data_dict[feature]
        except KeyError as e:
            raise KeyError("Feature {} not present in data".format(feature))

        values_for_symbols = values_for_features[symbol_list]

        return values_for_symbols.loc[current_datetime]
