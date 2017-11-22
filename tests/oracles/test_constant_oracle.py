import datetime
import os
from unittest import TestCase

import numpy as np

from delphi.scheduler.abstract_scheduler import SchedulingFrequency
from delphi.oracle.constant_oracle import ConstantOracle
from delphi.data_source.hdf5_data_source import StocksHDF5DataSource
from delphi.oracle.abstract_oracle import OracleActions


class TestConstantOracle(TestCase):

    @classmethod
    def setUpClass(cls):

        filename = os.path.join(os.path.dirname(__file__), '..', 'resources', '19990101_19990301_3_stocks.hdf5')

        data_source_config = {
            "filename": filename,
            "exchange": "NYSE",
            "start": datetime.datetime(1999, 1, 1),
            "end": datetime.datetime(1999, 3, 1)
        }

        cls.oracle_config = {
            "train_frequency": SchedulingFrequency.WEEKLY,
            "predict_frequency": SchedulingFrequency.DAILY,
            "predict_horizon": datetime.timedelta(days=1),
            "predict_offset": datetime.timedelta(minutes=30),
            "constant_variance": 0.1,
            "past_horizon": datetime.timedelta(days=7)
        }

        cls.data_source = StocksHDF5DataSource(data_source_config)

        cls.constant_oracle = ConstantOracle(cls.oracle_config)

    def test_single_predict(self):

        current_datetime = datetime.datetime(1999, 1, 20)

        event = OracleActions.PREDICT
        interval = self.constant_oracle.get_interval(event)

        data = self.data_source.get_data(current_datetime, interval)

        prediction = self.constant_oracle.predict(data, current_datetime)

        assert np.all([prediction.mean_vector.values == data['close'].iloc[-1].values])
