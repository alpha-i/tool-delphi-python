import os
import unittest

import pandas as pd
import numpy as np

from alphai_delphi import OraclePerformance
from alphai_delphi.performance.report import OracleReportWriter

from tests.performance import create_test_environment, destroy_test_environment, TMP_FOLDER


class TestReportWriter(unittest.TestCase):

    def setUp(self):
        create_test_environment()

    def tearDown(self):
        destroy_test_environment()

    def test_report_writer(self):

        output_path = TMP_FOLDER
        run_mode = 'backtest'

        performances = OraclePerformance(output_path, run_mode)

        sample_target_dt = pd.Timestamp("2017-05-30 15:44:00", tz='UTC')

        sample_symbols = ['AAPL', 'MSFT']
        n_symbols = len(sample_symbols)

        initial_values = pd.Series(np.array([1, 2]), index=sample_symbols)
        performances.add_initial_values(sample_target_dt, initial_values)

        final_values = pd.Series(np.array([3, 4]), index=sample_symbols)
        performances.add_final_values(sample_target_dt, final_values)

        mean_vector = pd.Series(np.array([1, 2]), index=sample_symbols)
        covariance_matrix = pd.DataFrame(np.identity(n_symbols), index=sample_symbols, columns=sample_symbols)
        performances.add_prediction(sample_target_dt, mean_vector, covariance_matrix)

        performances.save_to_hdf5(sample_target_dt)

        report_writer = OracleReportWriter(
            output_path,
            output_path,
            run_mode,
        )

        report_writer.write()

        expected_files = ["oracle_correlation_coefficient.pdf",
                          "oracle_cumulative_returns.pdf",
                          "oracle_data_table.csv",
                          "oracle_performance_table.csv",
                          "time-series-plot.pdf"
                          ]

        for filename in expected_files:
            assert os.path.isfile(os.path.join(output_path, filename))



