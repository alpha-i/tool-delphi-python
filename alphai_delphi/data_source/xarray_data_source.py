import datetime
import xarray as xr
import pytz
import time
import logging

from alphai_delphi.data_source.abstract_data_source import AbstractDataSource
from alphai_finance.data.cleaning import convert_to_utc, select_trading_hours
import pandas_market_calendars as mcal


def logtime(message=None):
    def wrap(method):
        def wrapped_f(*args, **kw):
            ts = time.time()
            result = method(*args, **kw)
            te = time.time()
            duration = te - ts
            msg = message if message else method.__name__
            # print("{}: Execution time: {} seconds".format(msg, duration))
            logging.info("{}: Execution time: {} seconds".format(msg, duration))
            return result

        return wrapped_f

    return wrap


class XArrayDataSource(AbstractDataSource):
    def __init__(self, configuration):
        super().__init__(configuration)
        self.data_timezone = pytz.timezone(self.config['data_timezone'])
        self._data = xr.open_dataset(self.config["filename"])
        self.symbols = self._data.data_vars.keys()
        self.calendar = mcal.get_calendar(self.config["exchange"])

    def start(self):
        return self.config["start"]

    def end(self):
        return self.config["end"]

    @logtime("get_data()")
    def get_data(self, current_datetime, interval):
        assert current_datetime.tzinfo == pytz.utc, "Datetime must provided in UTC timezone"
        start_datetime = current_datetime - interval
        end_datetime = current_datetime

        exchange_start_datetime = start_datetime.astimezone(self.data_timezone).replace(tzinfo=None)
        exchange_end_datetime = end_datetime.astimezone(self.data_timezone).replace(tzinfo=None)

        xray_data = self._data.sel(datetime=slice(exchange_start_datetime, exchange_end_datetime))
        data_dict = {
            'close': xray_data.sel(raw_feature='close').to_dataframe().drop(labels=["raw_feature"], axis=1),
            'volume': xray_data.sel(raw_feature='volume').to_dataframe().drop(labels=["raw_feature"], axis=1)
        }

        data_dict = convert_to_utc(data_dict)
        return select_trading_hours(data_dict, self.calendar)

    @logtime("values_for_symbols_feature_and_time()")
    def values_for_symbols_feature_and_time(self, symbol_list, feature, current_datetime):
        assert current_datetime.tzinfo == pytz.utc, "Datetime must provided in UTC timezone"
        current_datetime = current_datetime.astimezone(self.data_timezone).replace(tzinfo=None)
        start_datetime = current_datetime - datetime.timedelta(minutes=1)
        end_datetime = current_datetime
        values = self._data.sel(datetime=slice(start_datetime, end_datetime),
                                raw_feature=feature).to_dataframe()[symbol_list].loc[current_datetime]
        values.name = values.name.tz_localize(self.data_timezone).astimezone('UTC')
        return values


