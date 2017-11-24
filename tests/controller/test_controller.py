import os
import datetime
from tempfile import TemporaryDirectory

import pandas as pd
import pytz

from delphi.controller import Controller, ControllerConfiguration
from delphi.data_source.hdf5_data_source import StocksHDF5DataSource
from delphi.oracle import AbstractOracle
from delphi.oracle.abstract_oracle import PredictionResult
from delphi.oracle.constant_oracle import ConstantOracle
from delphi.oracle.oracle_configuration import OracleConfiguration
from delphi.scheduler import Scheduler


class DummyOracle(AbstractOracle):

    def get_interval(self, event):
        pass

    def save(self):
        pass

    def load(self):
        pass

    def predict(self, data, timestamp):
        return PredictionResult(
            pd.Series(),
            pd.DataFrame(),
            timestamp
        )

    def train(self, data):
        pass


filename = os.path.join(os.path.dirname(__file__), '..', 'resources', '19990101_19990301_3_stocks.hdf5')

temp_dir = TemporaryDirectory()


def test_controller_initialisation():

    exchange_name = "NYSE"
    simulation_start = datetime.datetime(1999, 1, 10, tzinfo=pytz.utc)
    simulation_end = datetime.datetime(1999, 2, 10, tzinfo=pytz.utc)

    data_source_config = {
        "filename": filename,
        "exchange": exchange_name,
        "start": datetime.datetime(1999, 1, 1, tzinfo=pytz.utc),
        "end": datetime.datetime(1999, 3, 1, tzinfo=pytz.utc)
    }

    datasource = StocksHDF5DataSource(data_source_config)
    oracle_config = OracleConfiguration(
        {
            "scheduling": {
                "prediction_horizon": 240,
                "prediction_frequency":
                    {
                        "frequency_type": 'DAILY',
                        "days_offset": 0,
                        "minutes_offset": 15
                    },
                "prediction_delta": 240,

                "training_frequency":
                    {
                        "frequency_type": 'WEEKLY',
                        "days_offset": 0,
                        "minutes_offset": 15
                    },
                "training_delta": 480,
            },
            "oracle": {
                "constant_variance": 0.1,
                "past_horizon": datetime.timedelta(days=7),
                "target_feature": 'close'
            }
        }
    )

    oracle = ConstantOracle(oracle_config)
    scheduler = Scheduler(simulation_start,
                          simulation_end,
                          exchange_name,
                          oracle.prediction_frequency,
                          oracle.training_frequency,
                          oracle.prediction_horizon
                          )

    controller_configuration = ControllerConfiguration({
        'performance_result_output': temp_dir.name,
        'start_date': simulation_start.strftime('%Y-%m-%d'),
        'end_date': simulation_end.strftime('%Y-%m-%d')
    })

    controller = Controller(
        configuration=controller_configuration,
        oracle=oracle,
        scheduler=scheduler,
        datasource=datasource
    )

    controller.run()
