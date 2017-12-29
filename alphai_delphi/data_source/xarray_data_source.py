import datetime
import xarray as xr
import pytz
import time

from alphai_delphi.data_source.abstract_data_source import AbstractDataSource


def logtime(message=None):
    def wrap(method):
        def wrapped_f(*args, **kw):
            ts = time.time()
            result = method(*args, **kw)
            te = time.time()
            duration = te - ts
            msg = message if message else method.__name__
            print("{}: Execution time: {} seconds".format(msg, duration))
            return result

        return wrapped_f

    return wrap


class XArrayDataSource(AbstractDataSource):
    def __init__(self, configuration):
        super().__init__(configuration)
        self.data_timezone = pytz.timezone(self.config['data_timezone'])
        self._data = xr.open_dataset(self.config["filename"])
        self.symbols = self._data.data_vars.keys()

    def start(self):
        return self.config["start"]

    def end(self):
        return self.config["end"]

    @logtime("get_data()")
    def get_data(self, current_datetime, interval):
        assert current_datetime.tzinfo == pytz.utc, "Datetime must provided in UTC timezone"
        current_datetime = current_datetime.astimezone(self.data_timezone).replace(tzinfo=None)
        start_datetime = current_datetime - interval
        end_datetime = current_datetime
        return self._data.sel(datetime=slice(start_datetime, end_datetime))

    @logtime("values_for_symbols_feature_and_time()")
    def values_for_symbols_feature_and_time(self, symbol_list, feature, current_datetime):
        assert current_datetime.tzinfo == pytz.utc, "Datetime must provided in UTC timezone"
        current_datetime = current_datetime.astimezone(self.data_timezone).replace(tzinfo=None)
        start_datetime = current_datetime - datetime.timedelta(minutes=1)
        end_datetime = current_datetime
        return self._data.sel(datetime=slice(start_datetime, end_datetime),
                              raw_features=feature).to_dataframe()[symbol_list].loc[current_datetime]


