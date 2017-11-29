import datetime
import glob
import os
import unittest
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest
import pytz

from delphi.controller import Controller, ControllerConfiguration
from delphi.data_source import AbstractDataSource
from delphi.data_source.hdf5_data_source import StocksHDF5DataSource
from delphi.oracle import AbstractOracle, PredictionResult
from delphi.oracle.constant_oracle import ConstantOracle
from delphi.oracle.oracle_configuration import OracleConfiguration
from delphi.oracle.performance import OraclePerformance
from delphi.scheduler import Scheduler

TEST_HDF5FILE_NAME = os.path.join(os.path.dirname(__file__), '..', 'resources', '19990101_19990301_3_stocks.hdf5')
TEMPORARY_DIRECTORY = TemporaryDirectory()


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


class DummyDataSource(AbstractDataSource):
    """
    A dummy data source, useful if you can't be bothered with the slow HDF5 source
    """

    @property
    def end(self):
        return datetime.datetime(1999, 1, 11, 14, 38)

    def get_data(self, current_datetime, interval):
        time_index = pd.DatetimeIndex(data=[
            pd.Timestamp('1999-01-11 14:33:00+00:00'),
            pd.Timestamp('1999-01-11 14:34:00+00:00'),
            pd.Timestamp('1999-01-11 14:35:00+00:00'),
            pd.Timestamp('1999-01-11 14:36:00+00:00'),
            pd.Timestamp('1999-01-11 14:37:00+00:00'),
            pd.Timestamp('1999-01-11 14:38:00+00:00')]
        )

        data = {
            'open': pd.DataFrame(index=time_index, columns=['AMZN', 'GOOG', 'MSFT'], data=np.random.random((6, 3))),
            'high': pd.DataFrame(index=time_index, columns=['AMZN', 'GOOG', 'MSFT'], data=np.random.random((6, 3))),
            'low': pd.DataFrame(index=time_index, columns=['AMZN', 'GOOG', 'MSFT'], data=np.random.random((6, 3))),
            'close': pd.DataFrame(index=time_index, columns=['AMZN', 'GOOG', 'MSFT'], data=np.random.random((6, 3))),
            'volume': pd.DataFrame(index=time_index, columns=['AMZN', 'GOOG', 'MSFT'], data=np.random.random((6, 3))),
        }
        return data

    @property
    def start(self):
        return datetime.datetime(1999, 1, 11, 14, 33)

    def values_for_symbols_feature_and_time(self, symbol_list, feature, current_datetime):
        return pd.DataFrame(data=np.random.random(3))


class TestController(unittest.TestCase):
    def test_controller_with_dummy_data_source(self):
        exchange_name = "NYSE"
        simulation_start = datetime.datetime(1999, 1, 10, tzinfo=pytz.utc)
        simulation_end = datetime.datetime(1999, 2, 10, tzinfo=pytz.utc)

        data_source_config = {
            "filename": TEST_HDF5FILE_NAME,
            "exchange": exchange_name,
            "start": datetime.datetime(1999, 1, 11, tzinfo=pytz.utc),
            "end": datetime.datetime(1999, 1, 11, tzinfo=pytz.utc)
        }

        datasource = DummyDataSource(data_source_config)
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
            'performance_result_output': TEMPORARY_DIRECTORY.name,
            'start_date': simulation_start.strftime('%Y-%m-%d'),
            'end_date': simulation_end.strftime('%Y-%m-%d')
        })

        oracle_performance = OraclePerformance(
            os.path.dirname(__file__), 'test'
        )

        controller = Controller(
            configuration=controller_configuration,
            oracle=oracle,
            scheduler=scheduler,
            datasource=datasource,
            performance=oracle_performance
        )

        controller.run()

        assert len(controller.prediction_results) == 14  # as the valid market days in the prediction range

        # Check if output files have been written
        assert len(
            glob.glob(os.path.dirname(__file__) + "/*hdf5")
        ) == 3

    # to run this test use add the parameter --runslow to the pytest invoker
    @pytest.mark.slow
    def test_controller_with_hdf5_data_source(self):
        exchange_name = "NYSE"
        simulation_start = datetime.datetime(1999, 1, 10, tzinfo=pytz.utc)
        simulation_end = datetime.datetime(1999, 2, 10, tzinfo=pytz.utc)

        data_source_config = {
            "filename": TEST_HDF5FILE_NAME,
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
                            "frequency_type": "DAILY",
                            "days_offset": 0,
                            "minutes_offset": 15
                        },
                    "prediction_delta": 10,

                    "training_frequency":
                        {
                            "frequency_type": "WEEKLY",
                            "days_offset": 0,
                            "minutes_offset": 15
                        },
                    "training_delta": 20,
                },
                "oracle": {
                    "constant_variance": 0.1,
                    "past_horizon": datetime.timedelta(days=7),
                    "target_feature": "close"
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
            'performance_result_output': TEMPORARY_DIRECTORY.name,
            'start_date': simulation_start.strftime('%Y-%m-%d'),
            'end_date': simulation_end.strftime('%Y-%m-%d')
        })

        oracle_performance = OraclePerformance(
            os.path.dirname(__file__), 'test'
        )

        controller = Controller(
            configuration=controller_configuration,
            oracle=oracle,
            scheduler=scheduler,
            datasource=datasource,
            performance=oracle_performance
        )

        controller.run()

        # Check if files have been writter
        assert len(glob.glob(os.path.dirname(__file__) + "/*hdf5")) == 3

    def tearDown(self):
        for output_file in glob.glob(os.path.dirname(__file__) + "/*hdf5"):
            os.remove(output_file)
