import datetime
from unittest import TestCase

import numpy as np
import pandas as pd
import pytz

from alphai_delphi.data_source.synthetic_data_source import SyntheticDataSource


class TestSyntheticDataSource(TestCase):

    @classmethod
    def setUpClass(cls):

        cls.config = {
            "start_date": datetime.datetime(1999, 1, 1),
            "end_date": datetime.datetime(1999, 3, 1)
        }

        cls.data_source = SyntheticDataSource(cls.config)

    def test_start(self):

        assert self.config["start_date"] == self.data_source.start

    def test_end(self):

        assert self.config["end_date"] == self.data_source.end

    def test_get_data(self):

        expected_symbols = ['sin_0', 'sin_1', 'sin_2']

        current_datetime = datetime.datetime(1999, 1, 15, 14, 31, tzinfo=pytz.utc)
        interval = datetime.timedelta(days=4)

        data_dict = self.data_source.get_data(current_datetime, interval)

        assert data_dict['close'].iloc[0].name.day == 11
        assert data_dict['close'].iloc[-1].name.day == 15

        assert np.all([data_dict['close'].columns == expected_symbols])

    def test_values_for_symbols_feature_and_time(self):
        expected_symbols = ['sin_0', 'sin_1', 'sin_2']

        current_datetime = datetime.datetime(1999, 1, 15, 14, 31, tzinfo=pytz.utc)

        data = self.data_source.values_for_symbols_feature_and_time(
            expected_symbols,
            'close',
            current_datetime
        )

        assert isinstance(data, pd.Series)
        assert set(data.index) == set(expected_symbols)

        expected_symbols = ['sin_0', 'sin_1']

        current_datetime = datetime.datetime(1999, 1, 15, 14, 31, tzinfo=pytz.utc)

        data = self.data_source.values_for_symbols_feature_and_time(
            expected_symbols,
            'close',
            current_datetime
        )

        assert isinstance(data, pd.Series)
        assert set(data.index) == set(expected_symbols)
