import datetime
import os
from unittest import TestCase

import numpy as np
import pandas as pd
import pytz

from delphi.data_source.hdf5_data_source import StocksHDF5DataSource


class TestHDF5DataProvider(TestCase):

    @classmethod
    def setUpClass(cls):

        filename = os.path.join(os.path.dirname(__file__), '..', 'resources', '19990101_19990301_3_stocks.hdf5')

        cls.config = {
            "filename": filename,
            "exchange": "NYSE",
            "start": datetime.datetime(1999, 1, 1),
            "end": datetime.datetime(1999, 3, 1)
        }

        cls.data_source = StocksHDF5DataSource(cls.config)

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
        assert data_dict['close'].iloc[-1].name.day == 15

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
        assert set(data.values) == {70.69, 11.769, 21.18}

        expected_symbols = ['AMZN', 'BAX']

        current_datetime = datetime.datetime(1999, 1, 15, 14, 30, tzinfo=pytz.utc)

        data = self.data_source.values_for_symbols_feature_and_time(
            expected_symbols,
            'close',
            current_datetime
        )

        assert isinstance(data, pd.Series)
        assert set(data.index) == set(expected_symbols)
        assert set(data.values) == {70.69, 11.769}





