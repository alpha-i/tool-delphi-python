import numpy as np
import pandas as pd
import pytz
from datetime import timedelta, datetime
from copy import deepcopy
import pandas_market_calendars as mcal
from delphi.data_source.abstract_data_source import AbstractDataSource
from alphai_time_series.make_series import random_walks


class StochasticProcessDataSource(AbstractDataSource):
    def __init__(self, configuration):
        super().__init__(configuration)
        self.timezone = pytz.utc
        self._data_dict = {}
        self._setup_data()

    def start(self):
        return self.config["start"]

    def end(self):
        return self.config["end"]

    def get_data(self, current_datetime, interval):
        assert current_datetime.tzinfo == self.timezone, "Datetime must provided in {} timezone".format(self.timezone)
        start_datetime = current_datetime - interval
        end_datetime = current_datetime

        data = {}
        for key in self._data_dict.keys():
            data[key] = self._data_dict[key][start_datetime:end_datetime]
        return data

    def values_for_symbols_feature_and_time(self, symbol_list, feature, current_datetime):
        if current_datetime.tzinfo != self.timezone:
            raise ValueError("Datetime must provided in {} timezone".format(self.timezone))
        try:
            values_for_features = self._data_dict[feature]
        except KeyError as e:
            raise KeyError("Feature {} not present in data".format(feature))

        values_for_symbols = values_for_features[symbol_list]

        return values_for_symbols.loc[current_datetime]


    @staticmethod
    def _create_minute_datetime_index(exchange_name, start_date, end_date):
        calendar = mcal.get_calendar(exchange_name)
        schedule = calendar.schedule(start_date, end_date)

        datetime_index = pd.DatetimeIndex([])

        for idx in range(len(schedule)):
            start_minute = schedule.market_open[idx]  # + timedelta(minutes=1) FIXME should we start at 931 or 930?
            end_minute = schedule.market_close[idx]
            datetime_index = datetime_index.append(pd.date_range(start=start_minute, end=end_minute, freq='min'))

        return datetime_index

    @staticmethod
    def _make_ohlcv_dict(dataframe):
        """

        :param dataframe: pandas dataframe
        :return: dict that with keys open high low close volume and the given dataframe as value
        """
        open = deepcopy(dataframe)  # FIXME change the var open -> open_vals

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

    def _setup_data(self):
        exchange_name = self.config['exchange']
        time_index = self._create_minute_datetime_index(exchange_name=exchange_name, start_date=self.start(),
                                                        end_date=self.end())
        correlation_coeff = 0.1
        offset = correlation_coeff / 5.
        n_series = 10
        variance = 1 / 1e3
        trend_period = -1
        mean_growth = 0
        noise_matrix = np.random.normal(loc=offset, scale=correlation_coeff, size=(n_series, n_series))
        noise_matrix = 0.5 * (noise_matrix + np.transpose(noise_matrix))   # FIXME is this correct mathematically?
        covariance = (np.eye(n_series) + noise_matrix) * variance
        n_timesteps = len(time_index)
        stochastic_process_output = random_walks(n_timesteps=n_timesteps, cov=covariance, trend_period=trend_period,
                                                 mean_growth=mean_growth)
        columns = ["walk_{}".format(clm) for clm in range(n_series)]
        stochastic_process_prices = pd.DataFrame(data=stochastic_process_output, index=time_index, columns=columns)

        self._data_dict = self._make_ohlcv_dict(stochastic_process_prices)

