import logging
import os
import warnings

import numpy as np
import pandas as pd
from tables import NaturalNameWarning

from alphai_delphi.performance.oracle import (
    create_oracle_performance_report,
    create_oracle_data_report,
    create_time_series_plot,
    read_oracle_results_from_path,
    read_oracle_symbol_weights_from_path
)

logger = logging.getLogger(__name__).addHandler(logging.NullHandler())
# We want to hide the number of NaturalNameWarning warnings when we format the column names
# according to the `TIMESTAMP_FORMAT`.
warnings.simplefilter(action='ignore', category=NaturalNameWarning)

ORACLE_RESULTS_MEAN_VECTOR_TEMPLATE = '{}_oracle_results_mean_vector.hdf5'
ORACLE_RESULTS_COVARIANCE_MATRIX_TEMPLATE = '{}_oracle_results_covariance_matrix.hdf5'
ORACLE_RESULTS_ACTUALS_TEMPLATE = '{}_oracle_results_actuals.hdf5'

METRIC_COLUMNS = ['returns_forecast_mean_vector', 'returns_forecast_covariance_matrix', 'initial_prices',
                  'final_prices', 'returns_actuals']

TIMESTAMP_FORMAT = '%Y%m%d-%H%M%S'


class OraclePerformance:
    def __init__(self, output_path, run_mode):
        self.run_mode = run_mode
        self.metrics = pd.DataFrame(columns=METRIC_COLUMNS)
        self._output_path = output_path
        self.output_mean_vector_filepath = \
            os.path.join(output_path, ORACLE_RESULTS_MEAN_VECTOR_TEMPLATE.format(run_mode))
        self.output_covariance_matrix_filepath = \
            os.path.join(output_path, ORACLE_RESULTS_COVARIANCE_MATRIX_TEMPLATE.format(run_mode))
        self.output_actuals_filepath = os.path.join(output_path, ORACLE_RESULTS_ACTUALS_TEMPLATE.format(run_mode))

    def add_prediction(self, target_dt, mean_vector, covariance_matrix):
        self.add_index_value(target_dt)
        self.metrics['returns_forecast_mean_vector'][target_dt] = mean_vector
        self.metrics['returns_forecast_covariance_matrix'][target_dt] = covariance_matrix

    def add_initial_prices(self, target_dt, initial_prices):
        self.add_index_value(target_dt)
        self.metrics['initial_prices'][target_dt] = initial_prices

    def add_final_values(self, target_dt, final_prices):
        if target_dt not in self.metrics.index:
            logger.error("Error in getting equity symbols at {}: target_dt not in index".format(target_dt))
        else:
            initial_prices = self.metrics.loc[target_dt, 'initial_prices']
            self.metrics['final_prices'][target_dt] = final_prices
            self.metrics['returns_actuals'][target_dt] = self.calculate_log_returns(initial_prices, final_prices)

    def get_symbols(self, target_dt):
        """
        Generalised form of get_equity_symbols

        :param target_dt: The datetime to get symbols at
        :type target_dt: datetime.datetime
        :return: an np.Array of symbols
        :rtype np.array
        """
        if target_dt not in self.metrics.index:
            logger.error("Error in getting equity symbols at {}: target_dt not in index".format(target_dt))
            return np.nan
        else:
            if isinstance(self.metrics.loc[target_dt, 'initial_prices'], pd.Series):
                return np.array(self.metrics.loc[target_dt, 'initial_prices'].index)
            elif not isinstance(self.metrics.loc[target_dt, 'returns_forecast_mean_vector'], pd.Series):
                return np.array(self.metrics.loc[target_dt, 'returns_forecast_mean_vector'].index)
            else:
                logger.error("Error in getting equity symbols at {}: no symbols could be found".format(target_dt))
                return np.nan

    @staticmethod
    def calculate_log_returns(initial_prices, final_prices):
        if set(initial_prices.index) != set(final_prices.index):
            logger.error("Can't calculate log returns: incompatibility between initial and final prices.")
            return np.nan
        else:
            return np.log(final_prices / initial_prices)

    def add_index_value(self, target_dt):
        if target_dt not in self.metrics.index:
            self.metrics = self.metrics.append(pd.DataFrame(index=[target_dt]))

    def save_to_hdf5(self, target_dt):
        if target_dt not in self.metrics.index or np.any(self.metrics.loc[target_dt, :].isnull()):
            logger.error("Failed to save to hdf5 at {}: target_dt not in index or nan was found".format(target_dt))
        else:
            target_dt_key = target_dt.strftime(format=TIMESTAMP_FORMAT)
            self.metrics.loc[target_dt, 'returns_forecast_mean_vector'].to_hdf(
                self.output_mean_vector_filepath, target_dt_key)
            self.metrics.loc[target_dt, 'returns_forecast_covariance_matrix'].to_hdf(
                self.output_covariance_matrix_filepath, target_dt_key)
            self.metrics.loc[target_dt, 'returns_actuals'].to_hdf(
                self.output_actuals_filepath, target_dt_key)

    def create_oracle_report(self):
        logger.info("Creating performance report...")
        results_path = self._output_path
        output_path = self._output_path
        oracle_results = read_oracle_results_from_path(results_path, run_mode=self.run_mode)
        oracle_symbol_weights = read_oracle_symbol_weights_from_path(results_path)
        create_oracle_performance_report(oracle_results, output_path, oracle_symbol_weights)
        create_oracle_data_report(oracle_results, output_path)
        create_time_series_plot(oracle_results, output_path)
        logger.info("Performance report finished.")

    def drop_dt(self, target_dt):
        if target_dt not in self.metrics.index:
            logger.error(
                "Could not drop target_dt = {}: target_dt not in index".format(target_dt))
        else:
            self.metrics = self.metrics.drop(target_dt)

    def __iter__(self):
        return iter(self.metrics.index)

    def __len__(self):
        return len(self.metrics)
