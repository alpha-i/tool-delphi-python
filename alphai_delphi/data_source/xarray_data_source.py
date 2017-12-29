import datetime
import xarray as xr
import pytz

from alphai_delphi.data_source.abstract_data_source import AbstractDataSource


class XArrayDataSource(AbstractDataSource):
    def __init__(self, configuration):
        super().__init__(configuration)
        self._data = xr.open_dataset(self.config["filename"])
        self.symbols = self._data.data_vars.keys()

    def start(self):
        return self.config["start"]

    def end(self):
        return self.config["end"]

    def get_data(self, current_datetime, interval):
        assert current_datetime.tzinfo == pytz.utc, "Datetime must provided in UTC timezone"
        current_datetime.replace(tzinfo=None)
        start_datetime = current_datetime - interval
        end_datetime = current_datetime
        return self._data.sel(datetime=slice(start_datetime, end_datetime))

    def values_for_symbols_feature_and_time(self, symbol_list, feature, current_datetime):
        assert current_datetime.tzinfo == pytz.utc, "Datetime must provided in UTC timezone"
        current_datetime.replace(tzinfo=None)
        start_datetime = current_datetime - datetime.timedelta(minutes=1)
        end_datetime = current_datetime
        return self._data.sel(datetime=slice(start_datetime, end_datetime)).get(symbol_list)


