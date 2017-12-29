"""
A file for running the Oracle
"""
import os
import datetime
import pytz
from alphai_delphi.oracle.constant_oracle import ConstantOracle
from alphai_delphi.controller import Controller, ControllerConfiguration
from alphai_delphi.data_source.stochastic_process_data_source import StochasticProcessDataSource
from alphai_delphi.data_source.xarray_data_source import XArrayDataSource
from alphai_delphi.scheduler import Scheduler
from alphai_delphi.oracle.oracle_configuration import OracleConfiguration
from alphai_delphi.performance.performance import OraclePerformance


def run_oracle():
    """
    A function to run the oracle
    """
    exchange_name = "NYSE"
    data_source_config = {
        "exchange": exchange_name,
        "filename": "/Users/tbs19/Documents/Data/Q_20061231_20111231_SP500_adjusted_1m_float32_close_volume_panel.nc",
        "start": datetime.datetime(2006, 12, 31, tzinfo=pytz.utc),
        "end": datetime.datetime(2011, 12, 31, tzinfo=pytz.utc)
    }
    datasource = XArrayDataSource(data_source_config)
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
    simulation_start = datetime.datetime(2009, 5, 1, tzinfo=pytz.utc)
    simulation_end = datetime.datetime(2009, 6, 1, tzinfo=pytz.utc)
    scheduler = Scheduler(
        simulation_start,
        simulation_end,
        exchange_name,
        oracle.prediction_frequency,
        oracle.training_frequency,
        oracle.prediction_horizon
    )
    controller_configuration = ControllerConfiguration(
        {
            'start_date': simulation_start.strftime('%Y-%m-%d'),
            'end_date': simulation_end.strftime('%Y-%m-%d')
        }
    )
    oracle_performance = OraclePerformance(
        os.path.join(os.path.dirname(__file__), "results"), 'test'
    )
    controller = Controller(
        configuration=controller_configuration,
        oracle=oracle,
        scheduler=scheduler,
        datasource=datasource,
        performance=oracle_performance
    )
    controller.run()


if __name__ == "__main__":
    run_oracle()
