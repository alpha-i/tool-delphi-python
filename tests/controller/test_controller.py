import datetime
import glob
import os
import unittest
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest
import pytz

from alphai_delphi.controller import Controller, ControllerConfiguration
from alphai_delphi.data_source import AbstractDataSource
from alphai_delphi.data_source.hdf5_data_source import StocksHDF5DataSource
from alphai_delphi.data_source.stochastic_process_data_source import StochasticProcessDataSource
from alphai_delphi.oracle.constant_oracle import ConstantOracle
from alphai_delphi.oracle.oracle_configuration import OracleConfiguration
from alphai_delphi.performance.performance import OraclePerformance
from alphai_delphi.scheduler import Scheduler

TEST_HDF5FILE_NAME = os.path.join(os.path.dirname(__file__), '..', 'resources', '19990101_19990301_3_stocks.hdf5')
TEMPORARY_DIRECTORY = TemporaryDirectory()


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
        time_index = pd.DatetimeIndex(data=[current_datetime])
        return pd.DataFrame(index=time_index, columns=['AMZN', 'GOOG', 'MSFT'],
                            data=[np.random.random(3)]).loc[current_datetime]


class TestController(unittest.TestCase):

    def test_get_market_interval(self):
        exchange_name = "NYSE"
        simulation_start = datetime.datetime(1999, 1, 10, tzinfo=pytz.utc)
        simulation_end = datetime.datetime(1999, 2, 10, tzinfo=pytz.utc)
        temp_dir = TemporaryDirectory()

        oracle_config = {
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

        controller = self._create_dummy_controller(exchange_name, simulation_end, simulation_start, temp_dir,
                                                   oracle_config)
        test_list = [
            dict(moment=pd.Timestamp("2018-01-29", tz=pytz.utc),
                 oracle_interval=datetime.timedelta(days=7),
                 expected_date=pd.Timestamp("2018-01-19", tz=pytz.utc)),

            dict(moment=pd.Timestamp("2017-12-27", tz=pytz.utc),
                 oracle_interval=datetime.timedelta(days=7),
                 expected_date=pd.Timestamp("2017-12-18", tz=pytz.utc)),

            dict(moment=pd.Timestamp("2018-01-27", tz=pytz.utc),
                 oracle_interval=datetime.timedelta(days=200),
                 expected_date=pd.Timestamp("2017-04-12", tz=pytz.utc)),
        ]

        for test in test_list:
            moment = test['moment']
            oracle_interval = test['oracle_interval']
            expected_date = test['expected_date']

            new_interval = controller.get_market_interval(moment, oracle_interval)
            new_date = moment - new_interval
            assert expected_date == new_date

    def test_controller_with_dummy_data_source(self):
        exchange_name = "NYSE"
        simulation_start = datetime.datetime(1999, 1, 10, tzinfo=pytz.utc)
        simulation_end = datetime.datetime(1999, 2, 10, tzinfo=pytz.utc)
        temp_dir = TemporaryDirectory()

        oracle_config = {
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

        controller = self._create_dummy_controller( exchange_name, simulation_end, simulation_start,
                                                    temp_dir, oracle_config)

        controller.run()

        assert len(controller.prediction_results) == 14  # as the valid market days in the prediction range

        # Check if output files have been written
        assert len(
            glob.glob(temp_dir.name + "/*hdf5")
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
            "data_timezone": "America/New_York",
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
            'start_date': simulation_start.strftime('%Y-%m-%d'),
            'end_date': simulation_end.strftime('%Y-%m-%d')
        })

        temp_dir = TemporaryDirectory()
        oracle_performance = OraclePerformance(
            temp_dir.name, 'test'
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
        assert len(glob.glob(temp_dir.name + "/*hdf5")) == 3

    # to run this test use add the parameter --runslow to the pytest invoker
    @pytest.mark.slow
    def test_controller_with_stochastic_process_data_source(self):
        exchange_name = "NYSE"
        data_source_config = {
            "exchange": exchange_name,
            "start": datetime.datetime(1999, 1, 1, tzinfo=pytz.utc),
            "end": datetime.datetime(1999, 3, 1, tzinfo=pytz.utc)
        }
        datasource = StochasticProcessDataSource(data_source_config)

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

        # these dates need to be within [start, end] of the data source
        simulation_start = datetime.datetime(1999, 1, 10, tzinfo=pytz.utc)
        simulation_end = datetime.datetime(1999, 2, 10, tzinfo=pytz.utc)
        scheduler = Scheduler(simulation_start,
                              simulation_end,
                              exchange_name,
                              oracle.prediction_frequency,
                              oracle.training_frequency,
                              oracle.prediction_horizon
                              )

        controller_configuration = ControllerConfiguration({
            'start_date': simulation_start.strftime('%Y-%m-%d'),
            'end_date': simulation_end.strftime('%Y-%m-%d')
        })

        temp_dir = TemporaryDirectory()
        oracle_performance = OraclePerformance(
            temp_dir.name, 'test'
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
        assert len(glob.glob(temp_dir.name + "/*hdf5")) == 3

    def _create_dummy_controller(self, exchange_name, simulation_end, simulation_start, temp_dir, oracle_config):
        data_source_config = {
            "filename": TEST_HDF5FILE_NAME,
            "exchange": exchange_name,
            "start": datetime.datetime(1999, 1, 11, tzinfo=pytz.utc),
            "end": datetime.datetime(1999, 1, 11, tzinfo=pytz.utc)
        }
        datasource = DummyDataSource(data_source_config)
        oracle_config = OracleConfiguration(oracle_config)

        oracle = ConstantOracle(oracle_config)
        scheduler = Scheduler(simulation_start,
                              simulation_end,
                              exchange_name,
                              oracle.prediction_frequency,
                              oracle.training_frequency,
                              oracle.prediction_horizon
                              )
        controller_configuration = ControllerConfiguration({
            'start_date': simulation_start.strftime('%Y-%m-%d'),
            'end_date': simulation_end.strftime('%Y-%m-%d')
        })
        oracle_performance = OraclePerformance(
            temp_dir.name, 'test'
        )
        controller = Controller(
            configuration=controller_configuration,
            oracle=oracle,
            scheduler=scheduler,
            datasource=datasource,
            performance=oracle_performance
        )
        return controller
