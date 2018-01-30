import datetime
import logging
from collections import namedtuple

import dateutils
import pandas_market_calendars as mcal
import pytz
from alphai_finance.data.cleaning import convert_to_utc, select_trading_hours
from alphai_finance.data.read_from_hdf5 import get_all_table_names_in_hdf5, read_feature_data_dict_from_hdf5

from alphai_delphi.data_source.abstract_data_source import AbstractDataSource
from alphai_delphi.data_source.utils import logtime

HDF5Cache = namedtuple('HDF5Cache', 'start end content')

logger = logging.getLogger(__name__)


class StocksHDF5DataSource(AbstractDataSource):
    def __init__(self, configuration):
        super().__init__(configuration)
        self.data_timezone = pytz.timezone(self.config['data_timezone'])
        self.filename = self.config["filename"]
        self.symbols = get_all_table_names_in_hdf5(self.filename)
        self.calendar = mcal.get_calendar(self.config["exchange"])

        # We'll hold a slice of the HDF5 contents here
        # to avoid reading it from the disk at every data request
        self._data_cache = None

    def start(self):
        return self.config["start"]

    def end(self):
        return self.config["end"]

    @logtime
    def get_data(self, current_datetime, interval):
        assert current_datetime.tzinfo == pytz.utc, "Datetime must provided in UTC timezone"
        start_datetime = current_datetime - interval
        end_datetime = current_datetime

        return self._extract_data(self.symbols, start_datetime, end_datetime)

    def _extract_data(self, symbols, start_datetime, end_datetime):
        exchange_start_datetime = start_datetime.astimezone(self.data_timezone).replace(tzinfo=None)
        exchange_end_datetime = end_datetime.astimezone(self.data_timezone).replace(tzinfo=None)

        data_start_time = exchange_start_datetime - datetime.timedelta(days=1)
        data_end_time = exchange_end_datetime

        if not self.interval_in_cache(data_start_time, data_end_time):
            logger.debug("Interval was NOT found in cache: %s - %s", data_start_time, data_end_time)
            self._data_cache = self.preload_year(data_start_time, data_end_time)
        else:
            logger.debug("Interval between %s and %s was found in cache", data_start_time, data_end_time)

        data_dict = {
            key: self._data_cache.content[key][data_start_time:data_end_time][symbols]
            for key in self._data_cache.content.keys()
        }
        data_dict = convert_to_utc(data_dict)
        return select_trading_hours(data_dict, self.calendar)

    def interval_in_cache(self, start, end):
        if not self._data_cache:
            return False
        return self._data_cache.start < start and self._data_cache.end > end

    def preload_year(self, start_time, end_time):
        del self._data_cache  # force the GC to dispose of old data
        preload_start = start_time - dateutils.relativedelta(months=2)  # because who knows?
        preload_end = end_time + dateutils.relativedelta(months=10)

        logger.debug("Preloading %s - %s", preload_start, preload_end)
        cache = HDF5Cache(
            start=preload_start,
            end=preload_end,
            content=read_feature_data_dict_from_hdf5(
                symbols=self.symbols,
                start=preload_start,
                end=preload_end,
                filepath=self.filename,
                timezone=self.data_timezone
            )
        )
        return cache

    @logtime
    def values_for_symbols_feature_and_time(self, symbol_list, feature, current_datetime):

        assert current_datetime.tzinfo == pytz.utc, "Datetime must provided in UTC timezone"

        data = self._extract_data(symbol_list, current_datetime - datetime.timedelta(minutes=1), current_datetime)
        try:
            values_for_features = data[feature]
        except KeyError as e:
            raise KeyError("Feature {} not present in data".format(feature))

        return values_for_features.loc[current_datetime]
