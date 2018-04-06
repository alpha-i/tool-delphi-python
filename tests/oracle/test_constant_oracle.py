import datetime
import os
from unittest import TestCase

import numpy as np
import pytz

from alphai_delphi.data_source.hdf5_data_source import StocksHDF5DataSource
from alphai_delphi.oracle.abstract_oracle import OracleAction
from alphai_delphi.oracle.constant_oracle import ConstantOracle


class TestConstantOracle(TestCase):

    @classmethod
    def setUpClass(cls):
        filename = os.path.join(os.path.dirname(__file__), '..', 'resources', '19990101_19990301_3_stocks.hdf5')

        calendar_name = "NYSE"
        data_source_config = {
            "filename": filename,
            "exchange": calendar_name,
            "start": datetime.datetime(1999, 1, 1, tzinfo=pytz.utc),
            "end": datetime.datetime(1999, 3, 1, tzinfo=pytz.utc),
            "data_timezone": "America/New_York"
        }

        scheduling_configuration = {
            "training_frequency": {
                "frequency_type": 'WEEKLY',
                "days_offset": 0,
                "minutes_offset": 30
            },
            "prediction_frequency": {
                "frequency_type": 'WEEKLY',
                "days_offset": 0,
                "minutes_offset": 30
            }
        }

        oracle_config = {
            "prediction_horizon": {
                "unit": "days",
                "value": 1
            },
            "prediction_delta": {
                'unit': 'days',
                'value': 10
            },
            "training_delta": {
                'unit': 'days',
                'value': 20
            },
            "model": {
                "constant_variance": 0.1,
                "past_horizon": datetime.timedelta(days=7),
                "target_feature": 'close'
            }
        }

        cls.data_source = StocksHDF5DataSource(data_source_config)

        cls.constant_oracle = ConstantOracle(
            calendar_name=calendar_name,
            oracle_configuration=oracle_config,
            scheduling_configuration=scheduling_configuration
        )

    def test_single_predict(self):
        current_datetime = datetime.datetime(1999, 1, 20, tzinfo=pytz.utc)

        event = OracleAction.PREDICT
        interval = self.constant_oracle.get_delta_for_event(event)

        data = self.data_source.get_data(current_datetime, interval)

        prediction = self.constant_oracle.predict(data, current_datetime)

        assert np.allclose(prediction.mean_vector.values, data['close'].iloc[-1].values, equal_nan=True)
        assert prediction.target_timestamp == current_datetime + self.constant_oracle.prediction_horizon
