import datetime
from unittest import TestCase

import numpy as np
import pandas as pd
import pytz

from alphai_delphi.data_source.stochastic_process_data_source import StochasticProcessDataSource


class TestStochasticProcessDataSource(TestCase):

    @classmethod
    def setUpClass(cls):

        cls.config = {
            "exchange": "NYSE",
            "start": datetime.datetime(1999, 1, 1),
            "end": datetime.datetime(1999, 3, 1)
        }

        cls.data_source = StochasticProcessDataSource(cls.config)

    def test_start(self):

        assert self.config["start"] == self.data_source.start

    def test_end(self):

        assert self.config["end"] == self.data_source.end

    def test_get_data(self):

        expected_symbols = ['walk_0', 'walk_1', 'walk_2', 'walk_3', 'walk_4', 'walk_5', 'walk_6', 'walk_7', 'walk_8',
                            'walk_9']

        current_datetime = datetime.datetime(1999, 1, 15, 14, 30, tzinfo=pytz.utc)
        interval = datetime.timedelta(days=4)

        data_dict = self.data_source.get_data(current_datetime, interval)

        assert data_dict['close'].iloc[0].name.day == 11
        assert data_dict['close'].iloc[-1].name.day == 15

        assert np.all([data_dict['close'].columns == expected_symbols])

    def test_values_for_symbols_feature_and_time(self):
        expected_symbols = ['walk_0', 'walk_1', 'walk_2', 'walk_3', 'walk_4', 'walk_5', 'walk_6', 'walk_7', 'walk_8',
                            'walk_9']

        current_datetime = datetime.datetime(1999, 1, 15, 14, 30, tzinfo=pytz.utc)

        data = self.data_source.values_for_symbols_feature_and_time(
            expected_symbols,
            'close',
            current_datetime
        )

        assert isinstance(data, pd.Series)
        assert set(data.index) == set(expected_symbols)

        expected_symbols = ['walk_0', 'walk_1']

        current_datetime = datetime.datetime(1999, 1, 15, 14, 30, tzinfo=pytz.utc)

        data = self.data_source.values_for_symbols_feature_and_time(
            expected_symbols,
            'close',
            current_datetime
        )

        assert isinstance(data, pd.Series)
        assert set(data.index) == set(expected_symbols)





