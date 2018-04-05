import os
import shutil
from tempfile import TemporaryDirectory
from unittest import TestCase, mock

import pandas as pd
import numpy as np
from numpy.testing import assert_almost_equal

from alphai_delphi.performance import DefaultMetrics, create_metric_filename
from alphai_delphi.performance.performance import OraclePerformance, TIMESTAMP_FORMAT

TMP_FOLDER = TemporaryDirectory().name


def create_test_environment():
    if not os.path.exists(TMP_FOLDER):
        os.makedirs(TMP_FOLDER)


def destroy_test_environment():
    shutil.rmtree(TMP_FOLDER)


class TestOraclePerformance(TestCase):
    def setUp(self):
        create_test_environment()
        self.output_path = TMP_FOLDER
        self.sample_target_dt = pd.Timestamp("2017-05-30 15:44:00", tz='UTC')
        self.sample_symbols = ['AAPL', 'MSFT']
        self.n_symbols = len(self.sample_symbols)
        self.sample_covariance_matrix_forecast = pd.DataFrame(np.identity(self.n_symbols),
                                                              index=self.sample_symbols, columns=self.sample_symbols)
        self.sample_mean_vector_forecast = pd.Series(np.zeros(self.n_symbols), index=self.sample_symbols)
        self.sample_initial_values = pd.Series(np.ones(self.n_symbols), index=self.sample_symbols)
        self.sample_log_return = 0.1
        self.sample_log_returns = pd.Series(self.sample_log_return * np.ones(self.n_symbols),
                                            index=self.sample_symbols)
        self.sample_final_values = pd.Series(np.exp(self.sample_log_return) * np.ones(self.n_symbols),
                                             index=self.sample_symbols)
        self.run_mode = 'backtest'

        self.oracle_perf = OraclePerformance(self.output_path, self.run_mode)

    def tearDown(self):
        destroy_test_environment()

    def test_add_prediction(self):
        self.oracle_perf.add_prediction(self.sample_target_dt,
                                        self.sample_mean_vector_forecast,
                                        self.sample_covariance_matrix_forecast)

        self.assertEqual(len(self.oracle_perf), 1)
        assert self.sample_target_dt in self.oracle_perf
        self.assertEqual(self.oracle_perf.metrics.index[0], self.sample_target_dt)
        assert self.sample_mean_vector_forecast.equals(
            self.oracle_perf.metrics.loc[self.sample_target_dt, DefaultMetrics.mean_vector.value])
        assert self.sample_covariance_matrix_forecast.equals(
            self.oracle_perf.metrics.loc[self.sample_target_dt, DefaultMetrics.covariance_matrix.value])

    def test_add_initial_values(self):
        self.oracle_perf.add_initial_values(self.sample_target_dt, self.sample_initial_values)

        self.assertEqual(len(self.oracle_perf), 1)
        assert self.sample_target_dt in self.oracle_perf
        self.assertEqual(self.oracle_perf.metrics.index[0], self.sample_target_dt)
        assert self.sample_initial_values.equals(
            self.oracle_perf.metrics.loc[self.sample_target_dt, 'initial_values'])

    @mock.patch('alphai_delphi.performance.performance.logger')
    def test_add_final_values(self, mock_logging):
        self.oracle_perf.add_final_values(self.sample_target_dt, self.sample_final_values)
        assert mock_logging.error.called

        self.oracle_perf.add_initial_values(self.sample_target_dt, self.sample_initial_values)
        self.oracle_perf.add_final_values(self.sample_target_dt, self.sample_final_values)
        self.assertEqual(len(self.oracle_perf), 1)
        assert self.sample_target_dt in self.oracle_perf
        self.assertEqual(self.oracle_perf.metrics.index[0], self.sample_target_dt)
        assert self.sample_final_values.equals(
            self.oracle_perf.metrics.loc[self.sample_target_dt, DefaultMetrics.final_values.value])
        assert_almost_equal(self.sample_log_returns.values,
                            self.oracle_perf.metrics.loc[
                                self.sample_target_dt, DefaultMetrics.returns_actuals.value].values)

    @mock.patch('alphai_delphi.performance.performance.logger')
    def test_get_equity_symbols(self, mock_logging):
        self.oracle_perf.get_symbols(self.sample_target_dt)
        assert mock_logging.error.called

        self.oracle_perf.add_initial_values(self.sample_target_dt, self.sample_initial_values)
        symbols = self.oracle_perf.get_symbols(self.sample_target_dt)
        assert set(symbols) == set(self.sample_symbols)

    def test_add_index_value(self):
        self.oracle_perf.add_index_value(self.sample_target_dt)
        self.assertEqual(len(self.oracle_perf), 1)
        assert self.sample_target_dt in self.oracle_perf

        self.oracle_perf.add_index_value(self.sample_target_dt)
        self.assertEqual(len(self.oracle_perf), 1)
        assert self.sample_target_dt in self.oracle_perf

    def test_add_custom_metric_and_save(self):
        self.oracle_perf.add_initial_values(self.sample_target_dt, self.sample_initial_values)
        self.oracle_perf.add_prediction(self.sample_target_dt,
                                        self.sample_mean_vector_forecast,
                                        self.sample_covariance_matrix_forecast)
        self.oracle_perf.add_final_values(self.sample_target_dt, self.sample_final_values)

        custom_metric = 'custom_metric'
        custom_metric_value = 1

        self.oracle_perf.add_metric(custom_metric, custom_metric_value, self.sample_target_dt)
        self.oracle_perf.save_to_hdf5(self.sample_target_dt)

        path = os.path.join(self.output_path, create_metric_filename(custom_metric, self.run_mode))
        assert os.path.isfile(path)
        with pd.HDFStore(path) as store:
            target_dt_key = self.sample_target_dt.strftime(format=TIMESTAMP_FORMAT)
            values = store.get(target_dt_key)
            assert values.equals(pd.Series(custom_metric_value))

    def test_save_to_hdf5(self):
        self.oracle_perf.add_initial_values(self.sample_target_dt, self.sample_initial_values)
        self.oracle_perf.add_prediction(self.sample_target_dt,
                                        self.sample_mean_vector_forecast,
                                        self.sample_covariance_matrix_forecast)
        self.oracle_perf.add_final_values(self.sample_target_dt, self.sample_final_values)
        self.oracle_perf.save_to_hdf5(self.sample_target_dt)

        second_target_dt = pd.Timestamp("2017-05-30 18:44:00", tz='UTC')
        self.oracle_perf.add_initial_values(second_target_dt, self.sample_initial_values)
        self.oracle_perf.add_prediction(second_target_dt,
                                        self.sample_mean_vector_forecast,
                                        self.sample_covariance_matrix_forecast)
        self.oracle_perf.add_final_values(second_target_dt, self.sample_final_values)
        self.oracle_perf.save_to_hdf5(second_target_dt)

        expected_files = []
        for metric_name in self.oracle_perf._output_metrics:
            path = os.path.join(self.output_path, create_metric_filename(metric_name, self.run_mode))
            assert os.path.isfile(path)
            expected_files.append((metric_name, path))

        for metric_name, path in expected_files:
            with pd.HDFStore(path) as store:
                for target_dt in self.oracle_perf:
                    target_dt_key = target_dt.strftime(format=TIMESTAMP_FORMAT)
                    values = store.get(target_dt_key)
                    assert values.equals(self.oracle_perf.metrics.loc[target_dt, metric_name])

    @mock.patch('alphai_delphi.performance.performance.logger')
    def test_drop_dt(self, mock_logging):
        self.oracle_perf.add_initial_values(self.sample_target_dt, self.sample_initial_values)
        self.oracle_perf.add_prediction(self.sample_target_dt,
                                        self.sample_mean_vector_forecast,
                                        self.sample_covariance_matrix_forecast)
        self.oracle_perf.add_final_values(self.sample_target_dt, self.sample_final_values)
        assert self.sample_target_dt in self.oracle_perf

        self.oracle_perf.drop_dt(self.sample_target_dt)
        assert self.sample_target_dt not in self.oracle_perf
        assert not mock_logging.error.called

        self.oracle_perf.drop_dt(self.sample_target_dt)
        assert mock_logging.error.called
