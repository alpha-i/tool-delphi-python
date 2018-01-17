import os
import datetime
import numpy as np
import pytz
import pandas as pd
from unittest import TestCase

from alphai_delphi.data_source.xarray_data_source import XArrayDataSource


class TestXArrayDataSource(TestCase):

    @classmethod
    def setUpClass(cls):

        filename = os.path.join(os.path.dirname(__file__), '..', 'resources', '19990101_19990301_3_stocks.nc')

        cls.config = {
            "filename": filename,
            "exchange": "NYSE",
            "data_timezone": "America/New_York",
            "start": datetime.datetime(1999, 1, 1),
            "end": datetime.datetime(1999, 3, 1)
        }

        cls.data_source = XArrayDataSource(cls.config)

    def test_start(self):

        assert self.config["start"] == self.data_source.start()

    def test_end(self):

        assert self.config["end"] == self.data_source.end()

    def test_get_data(self):

        expected_symbols = ['AMZN', 'BAX', 'CSCO']

        current_datetime = datetime.datetime(1999, 1, 15, 14, 30, tzinfo=pytz.utc)
        interval = datetime.timedelta(days=4)

        data_dict = self.data_source.get_data(current_datetime, interval)

        assert data_dict['close'].iloc[0].name.day == 11
        assert data_dict['close'].iloc[0].name.hour == 14
        assert data_dict['close'].iloc[0].name.minute == 30
        assert data_dict['close'].iloc[0].name.tzinfo == pytz.utc

        assert data_dict['close'].iloc[-1].name.day == 15
        assert data_dict['close'].iloc[0].name.hour == 14
        assert data_dict['close'].iloc[0].name.minute == 30
        assert data_dict['close'].iloc[0].name.tzinfo == pytz.utc

        assert np.all([data_dict['close'].columns == expected_symbols])

    def test_values_for_symbols_feature_and_time(self):
        expected_symbols = ['AMZN', 'BAX', 'CSCO']

        current_datetime = datetime.datetime(1999, 1, 15, 14, 30, tzinfo=pytz.utc)

        data = self.data_source.values_for_symbols_feature_and_time(
            expected_symbols,
            'close',
            current_datetime
        )

        assert isinstance(data, pd.Series)
        assert set(data.index) == set(expected_symbols)
        np.testing.assert_almost_equal(list(data.values), [70.19, np.nan,  20.81])

        expected_symbols = ['AMZN', 'BAX']

        current_datetime = datetime.datetime(1999, 1, 15, 14, 30, tzinfo=pytz.utc)

        data = self.data_source.values_for_symbols_feature_and_time(
            expected_symbols,
            'close',
            current_datetime
        )

        assert isinstance(data, pd.Series)
        assert set(data.index) == set(expected_symbols)
        np.testing.assert_almost_equal(list(data.values), [70.19, np.nan])
