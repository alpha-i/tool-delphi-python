import logging
import os
import warnings

import numpy as np
import pandas as pd
from tables import NaturalNameWarning

from alphai_delphi.performance import DefaultMetrics, create_metric_filename
from alphai_delphi.performance.report import OracleReportWriter

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# We want to hide the number of NaturalNameWarning warnings when we format the column names
# according to the `TIMESTAMP_FORMAT`.
warnings.simplefilter(action='ignore', category=NaturalNameWarning)

TIMESTAMP_FORMAT = '%Y%m%d-%H%M%S'


class OraclePerformance:
    def __init__(self, output_path, run_mode):
        self.run_mode = run_mode
        self.metrics = pd.DataFrame(columns=DefaultMetrics.get_metrics())
        self._output_path = output_path
        self._output_metrics = [
                DefaultMetrics.mean_vector.value,
                DefaultMetrics.covariance_matrix.value,
                DefaultMetrics.returns_actuals.value
        ]

    def add_prediction(self, target_dt, mean_vector, covariance_matrix):
        self.add_index_value(target_dt)
        self.metrics[DefaultMetrics.mean_vector.value][target_dt] = mean_vector
        self.metrics[DefaultMetrics.covariance_matrix.value][target_dt] = covariance_matrix

    def add_initial_values(self, target_dt, initial_values):
        self.add_index_value(target_dt)
        self.metrics[DefaultMetrics.initial_values.value][target_dt] = initial_values

    def add_final_values(self, target_dt, final_values):
        if target_dt not in self.metrics.index:
            logger.error("Error in getting equity symbols at {}: target_dt not in index".format(target_dt))
        else:
            initial_values = self.metrics.loc[target_dt, DefaultMetrics.initial_values.value]
            self.metrics[DefaultMetrics.final_values.value][target_dt] = final_values
            self.metrics[DefaultMetrics.returns_actuals.value][target_dt] = self.calculate_log_returns(initial_values,
                                                                                                       final_values)

    def add_features_sensitivity(self, target_dt, features_sensitivity):
        self.add_index_value(target_dt)
        if 'features_sensitivity' not in self.metrics.columns:
            self.metrics['features_sensitivity'] = np.object
        self.metrics['features_sensitivity'][target_dt] = pd.Series(features_sensitivity)

    def get_symbols(self, target_dt):
        """
        Generalised form of get_equity_symbols

        :param target_dt: The datetime to get symbols at
        :type target_dt: datetime.datetimeq
        :return: an np.Array of symbols
        :rtype np.array
        """
        if target_dt not in self.metrics.index:
            logger.error("Error in getting equity symbols at {}: target_dt not in index".format(target_dt))
            return np.nan
        else:
            if isinstance(self.metrics.loc[target_dt, DefaultMetrics.initial_values.value], pd.Series):
                return np.array(self.metrics.loc[target_dt, DefaultMetrics.initial_values.value].index)
            elif not isinstance(self.metrics.loc[target_dt, DefaultMetrics.mean_vector.value], pd.Series):
                return np.array(self.metrics.loc[target_dt, DefaultMetrics.mean_vector.value].index)
            else:
                logger.error("Error in getting equity symbols at {}: no symbols could be found".format(target_dt))
                return np.nan

    @staticmethod
    def calculate_log_returns(initial_values, final_values):
        if set(initial_values.index) != set(final_values.index):
            logger.error("Can't calculate log returns: incompatibility between initial and final prices.")
            return np.nan
        else:
            return np.log(final_values / initial_values)

    def add_index_value(self, target_dt):
        if target_dt not in self.metrics.index:
            self.metrics = self.metrics.append(pd.DataFrame(index=[target_dt]))

    def save_to_hdf5(self, target_dt):
        if target_dt not in self.metrics.index or np.any(self.metrics.loc[target_dt, :].isnull()):
            logger.error("Failed to save to hdf5 at {}: target_dt not in index or nan was found".format(target_dt))
        else:

            for metric in self._output_metrics:
                self._save_metric_to_hdf(metric, target_dt)

            if 'features_sensitivity' in self.metrics.columns:
                target_dt_key = target_dt.strftime(format=TIMESTAMP_FORMAT)
                self.metrics.loc[target_dt, 'features_sensitivity'].to_hdf(
                    self.output_feature_sensitivity_filepath, target_dt_key
                )

    def _save_metric_to_hdf(self, metric_name, target_datetime):

        target_dt_key = target_datetime.strftime(format=TIMESTAMP_FORMAT)
        path = os.path.join(self._output_path, create_metric_filename(metric_name, self.run_mode))
        self.metrics.loc[target_datetime, metric_name].to_hdf(path, target_dt_key)

    def create_oracle_report(self):
        report = OracleReportWriter(self._output_path, self._output_path, self.run_mode)
        report.write()

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
