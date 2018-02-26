import logging
import os

import matplotlib

matplotlib.use('Agg')

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import (
    MaxNLocator,
    FuncFormatter,
)
import seaborn as sns

from alphai_time_series.performance_trials.calculator import Calculator

sns.set_context("talk")

logger = logging.getLogger(__name__)

INFLATION_FACTOR = 1.1
DEFLATION_FACTOR = 0.9
N_EXTREME_SERIES = 10
EPSILON = 1e-12  # Zeros cause problems with binary accuracy
DEFAULT_MAX_N_TICKS = 10
N_ROLLING_WINDOW = 20

ORACLE_MEAN_VECTOR_ENDSWITH_TEMPLATE = 'oracle_results_mean_vector.hdf5'
ORACLE_COVARIANCE_MATRIX_ENDSWITH_TEMPLATE = 'oracle_results_covariance_matrix.hdf5'
ORACLE_ACTUALS_ENDSWITH_TEMPLATE = 'oracle_results_actuals.hdf5'
ORACLE_SYMBOL_WEIGHTS_ENDSWITH_TEMPLATE = 'oracle_symbol_weights.csv'
FINANCIAL_RETURNS_ENDSWITH_TEMPLATE = 'financial_returns.csv'
FINANCIAL_POSITIONS_ENDSWITH_TEMPLATE = 'financial_positions.csv'
FINANCIAL_TRANSACTIONS_ENDSWITH_TEMPLATE = 'financial_transactions.csv'
FINANCIAL_BENCHMARK_ENDSWITH_TEMPLATE = 'benchmark.csv'
BENCHMARK_NAME = 'SPY'
ORACLE_METRIC_COLUMNS = ['returns_forecast_mean_vector', 'returns_forecast_covariance_matrix', 'returns_actuals']


def create_oracle_performance_report(oracle_results, output_path, oracle_symbol_weights=None):
    """
    Calculate oracle performance metrics and save the table to csv
    :param oracle_results: Dataframe indexed by datetime with three columns
    ['returns_actuals', 'returns_forecast_covariance_matrix', 'returns_forecast_mean_vector']
    :param output_path: path where to save the output
    :param oracle_symbol_weights:
    :return: Nothing
    """
    calc = Calculator()
    oracle_performance_table = pd.Series()
    n_samples = len(oracle_results)
    chi_squared_array = pd.Series()
    log_likelihood_array = pd.Series()
    optimal_chi_squared_array = pd.Series()
    optimal_log_likelihood_array = pd.Series()
    null_chi_squared_array = pd.Series()
    null_log_likelihood_array = pd.Series()
    inverted_chi_squared_array = pd.Series()
    inverted_log_likelihood_array = pd.Series()
    inflated_cov_chi_squared_array = pd.Series()
    inflated_cov_log_likelihood_array = pd.Series()
    deflated_cov_chi_squared_array = pd.Series()
    deflated_cov_log_likelihood_array = pd.Series()
    corr_coeff = pd.Series()
    weighted_corr_coeff = pd.Series()
    frac_change_portfolio = pd.Series()
    frac_change_market = pd.Series()
    binary_accumulator = 0
    extreme_fall_binary_accumulator = 0
    extreme_rise_binary_accumulator = 0
    true_win_loss_fraction_accumulator = 0
    forecast_win_loss_fraction_accumulator = 0
    n_total_samples = 0
    n_total_extreme_samples = 0
    absolute_binary_accumulator = 0

    valid_covariances = check_validity_of_covariances(oracle_results)

    if valid_covariances:
        inv_covariance_matrices = compute_covariance_inverses(oracle_results)
        logger.info("Attempting to compute optimal covariance matrix")
        cov_factor = optimise_covariance(oracle_results, inv_covariance_matrices)
    else:
        cov_factor = 1
        inv_covariance_matrices = []
        for i in range(n_samples):
            date = oracle_results.index[i]
            forecast = oracle_results.returns_forecast_mean_vector[date].values
            n_predictions = len(forecast)
            diag_matrix = np.eye(n_predictions)
            inv_covariance_matrices.append(diag_matrix)
            oracle_results.returns_forecast_covariance_matrix[date] = diag_matrix

    logger.info("Evaluating likelihoods")
    for i in range(n_samples):
        date_str = oracle_results.index[i]
        date = pd.datetime.strptime(date_str, '%Y%m%d-%H%M%S').date()

        pd_truth = oracle_results.returns_actuals[date_str]
        pd_forecast = oracle_results.returns_forecast_mean_vector[date_str]

        true_cols = list(pd_truth.index.values)
        forecast_cols = list(pd_forecast.index.values)
        assert true_cols == forecast_cols, "List of symbols that don't match"

        truth = np.asarray(pd_truth.values)
        forecast = np.asarray(pd_forecast.values)

        if oracle_symbol_weights is None:
            weights = None
        else:
            predicted_symbols = pd_truth.index.tolist()
            weights = extract_weight_array(predicted_symbols, oracle_symbol_weights)
        weighted_corr_coeff.loc[date] = calculate_weighted_correlation_coefficient(truth, forecast, weights)

        covariance = oracle_results.returns_forecast_covariance_matrix[date_str]
        covariance = np.asarray(covariance)
        optimal_covariance = cov_factor * covariance

        truth, forecast, covariance, masked_optimal_covariance = \
            calc_masked_forecasts(truth, forecast, covariance, optimal_covariance)

        forecast_correlation_matrix = np.corrcoef(truth, forecast)
        corr_coeff.loc[date] = forecast_correlation_matrix[0, 1]

        chi_squared_array.loc[date] = calc.chi_squared(truth, forecast, covariance)

        log_likelihood_array.loc[date] = calc.log_likelihood(chi_squared_array.loc[date], covariance)

        optimal_chi_squared_array.loc[date] = calc.chi_squared(truth, forecast, masked_optimal_covariance)
        optimal_log_likelihood_array.loc[date] = calc.log_likelihood(optimal_chi_squared_array.loc[date],
                                                                     masked_optimal_covariance)

        null_chi_squared_array.loc[date] = calc.chi_squared(truth, 0 * forecast, covariance)
        null_log_likelihood_array.loc[date] = calc.log_likelihood(null_chi_squared_array.loc[date], covariance)

        inverted_chi_squared_array.loc[date] = calc.chi_squared(truth, -1 * forecast, covariance)
        inverted_log_likelihood_array.loc[date] = calc.log_likelihood(inverted_chi_squared_array.loc[date], covariance)

        inflated_cov = INFLATION_FACTOR * covariance
        inflated_cov_chi_squared_array.loc[date] = calc.chi_squared(truth, forecast, inflated_cov)
        inflated_cov_log_likelihood_array.loc[date] = calc.log_likelihood(inflated_cov_chi_squared_array.loc[date],
                                                                          inflated_cov)

        deflated_cov = DEFLATION_FACTOR * covariance
        deflated_cov_chi_squared_array.loc[date] = calc.chi_squared(truth, forecast, deflated_cov)
        deflated_cov_log_likelihood_array.loc[date] = calc.log_likelihood(deflated_cov_chi_squared_array.loc[date],
                                                                          deflated_cov)

        if i == 0:
            logger.info('Example truth: {}.'.format(truth[1:10]))
            logger.info('Example forecast: {}.'.format(forecast[1:10]))

        true_relative_winners_losers = calculate_winners_losers(truth)
        forecast_relative_winners_losers = calculate_winners_losers(forecast)

        binary_accumulator += np.mean(true_relative_winners_losers == forecast_relative_winners_losers)

        true_win_loss_fraction_accumulator += np.mean(np.sign(truth + EPSILON) == 1)
        forecast_win_loss_fraction_accumulator += np.mean(np.sign(forecast) == 1)

        absolute_binary_accumulator += np.mean(np.sign(truth) == np.sign(forecast))

        n_series = len(forecast)
        if N_EXTREME_SERIES >= n_series:
            n_extreme_series = 1
        else:
            n_extreme_series = N_EXTREME_SERIES

        # Find indices of the largest forecasts
        extreme_win_index = np.argpartition(forecast, -n_extreme_series)[-n_extreme_series:]
        extreme_truth = true_relative_winners_losers[extreme_win_index]
        extreme_forecast = forecast_relative_winners_losers[extreme_win_index]
        extreme_fall_binary_accumulator += np.mean(np.sign(extreme_truth) == np.sign(extreme_forecast))

        extreme_lose_index = np.argpartition(-forecast, -n_extreme_series)[-n_extreme_series:]
        extreme_truth = true_relative_winners_losers[extreme_lose_index]
        extreme_forecast = forecast_relative_winners_losers[extreme_lose_index]
        extreme_rise_binary_accumulator += np.mean(np.sign(extreme_truth) == np.sign(extreme_forecast))

        long_returns = truth[extreme_win_index]
        short_returns = - truth[extreme_lose_index]
        portfolio_change = np.concatenate((long_returns, short_returns)).flatten()
        frac_change_portfolio.loc[date] = np.nanmean(portfolio_change)
        frac_change_market.loc[date] = np.nanmean(truth)

        n_total_samples += len(truth)
        n_total_extreme_samples += len(extreme_truth)

    total_portfolio_return = np.exp(np.nansum(frac_change_portfolio))
    total_market_return = np.exp(np.nansum(frac_change_market))

    oracle_performance_table["rms"] = np.nanstd(np.concatenate(oracle_results.returns_actuals, axis=0) -
                                                np.concatenate(oracle_results.returns_forecast_mean_vector, axis=0))

    oracle_performance_table["min_rms"] = np.nanstd(np.concatenate(oracle_results.returns_actuals, axis=0))

    oracle_performance_table["Median Correlation Coeff"] = np.nanmedian(corr_coeff)
    oracle_performance_table["Median Wt Correlation Coeff"] = np.nanmedian(weighted_corr_coeff)

    oracle_performance_table["total_financial_return"] = total_portfolio_return
    oracle_performance_table["financial_return_relative_to_market"] = total_portfolio_return / total_market_return

    n_nans = np.isnan(chi_squared_array).sum()
    n_infs = np.isinf(chi_squared_array).sum()

    total_log_l, log_l_per_samples = calc_masked_likelihoods(log_likelihood_array)
    null_log_l, null_log_l_per_samples = calc_masked_likelihoods(null_log_likelihood_array)
    inv_total_log_l, inv_log_l_per_samples = calc_masked_likelihoods(inverted_log_likelihood_array)
    inf_total_log_l, inf_log_l_per_samples = calc_masked_likelihoods(inflated_cov_log_likelihood_array)
    def_total_log_l, def_log_l_per_samples = calc_masked_likelihoods(deflated_cov_log_likelihood_array)
    max_total_log_l, max_log_l_per_samples = calc_masked_likelihoods(optimal_log_likelihood_array)

    oracle_performance_table['NaNs found in chi2'] = n_nans
    oracle_performance_table['Infs found in chi2'] = n_infs

    oracle_performance_table["reduced-chi2"] = calc_masked_reduced_chi2(chi_squared_array)
    oracle_performance_table['total-log-likelihood'] = total_log_l
    oracle_performance_table['log-likelihood-per-sample'] = log_l_per_samples

    oracle_performance_table['n-total-samples'] = n_total_samples
    oracle_performance_table['n-valid-samples'] = total_log_l / log_l_per_samples

    oracle_performance_table['optimal_cov_factor'] = cov_factor
    oracle_performance_table['max-total-log-likelihood'] = max_total_log_l
    oracle_performance_table['max-log-likelihood-per-sample'] = max_log_l_per_samples

    oracle_performance_table["null-reduced-chi2"] = calc_masked_reduced_chi2(null_chi_squared_array)
    oracle_performance_table['null-total-log-likelihood'] = null_log_l
    oracle_performance_table['null-log-likelihood-per-sample'] = null_log_l_per_samples

    oracle_performance_table["inverted-reduced-chi2"] = calc_masked_reduced_chi2(inverted_chi_squared_array)
    oracle_performance_table['inverted-total-log-likelihood'] = inv_total_log_l
    oracle_performance_table['inverted-log-likelihood-per-sample'] = inv_log_l_per_samples

    oracle_performance_table["inflated-cov-reduced-chi2"] = calc_masked_reduced_chi2(inflated_cov_chi_squared_array)
    oracle_performance_table['inflated-cov-total-log-likelihood'] = inf_total_log_l
    oracle_performance_table['inflated-cov-log-likelihood-per-sample'] = inf_log_l_per_samples

    oracle_performance_table["deflated-cov-reduced-chi2"] = calc_masked_reduced_chi2(deflated_cov_chi_squared_array)
    oracle_performance_table['deflated-cov-total-log-likelihood'] = def_total_log_l
    oracle_performance_table['deflated-cov-log-likelihood-per-sample'] = def_log_l_per_samples

    oracle_performance_table['binary-accuracy'] = binary_accumulator / n_samples
    oracle_performance_table['binary-accuracy-precision'] = 1 / np.sqrt(n_total_samples)

    oracle_performance_table['extreme-rise-binary-accuracy'] = extreme_rise_binary_accumulator / n_samples
    oracle_performance_table['extreme-rise-binary-accuracy-precision'] = 1 / np.sqrt(n_total_extreme_samples)
    oracle_performance_table['extreme-fall-binary-accuracy'] = extreme_fall_binary_accumulator / n_samples
    oracle_performance_table['extreme-fall-binary-accuracy-precision'] = 1 / np.sqrt(n_total_extreme_samples)

    oracle_performance_table['true-win-loss-frac'] = true_win_loss_fraction_accumulator / n_samples
    oracle_performance_table['forecast-win-loss-frac'] = forecast_win_loss_fraction_accumulator / n_samples

    oracle_performance_table['absolute-binary-accuracy'] = absolute_binary_accumulator / n_samples

    oracle_performance_table.to_csv(os.path.join(output_path, 'oracle_performance_table.csv'))

    # Plots
    create_moving_average_figure(
        corr_coeff,
        'Correlation Coefficient',
        N_ROLLING_WINDOW
    ).savefig(os.path.join(output_path, 'oracle_correlation_coefficient.pdf'))

    create_time_series_comparison_figure(
        [frac_change_portfolio.cumsum(), frac_change_market.cumsum()],
        ['Portfolio', 'Market'], 'Cumulative returns'
    ).savefig(os.path.join(output_path, 'oracle_cumulative_returns.pdf'))


def optimise_covariance(oracle_results, inv_covariance_matrices):
    """ Searches for some optimal rescaling of the cov matrix.
    """

    optimal_cov_factor = minimize_scalar(
        oracle_covariance_optimisation,
        args=(oracle_results, inv_covariance_matrices),
        bounds=(1e-3, 1e3), method='bounded'
    )
    logger.info("Optimizer results: {}".format(optimal_cov_factor))

    return optimal_cov_factor.x


def oracle_covariance_optimisation(cov_factor, oracle_results, inv_covariance_matrices):
    """ Provides objective to minimise when looking for optimal covariance matrix. """

    cost = 0
    n_samples = len(oracle_results)

    for i in range(n_samples):
        date = oracle_results.index[i]
        truth = oracle_results.returns_actuals[date]
        forecast = oracle_results.returns_forecast_mean_vector[date]
        inv_covariance = inv_covariance_matrices[i]
        sample_cost = neg_log_likelihood(cov_factor, truth, forecast, inv_covariance)

        if not np.isnan(sample_cost):
            cost += sample_cost

    return cost


def neg_log_likelihood(cov_factor, truth, forecast, inv_cov):
    """ Provides the target to minmise (related to negative log likelihood; ignoring constant contributions) """

    n_dims = inv_cov.shape[0]
    scaled_log_det = n_dims * np.log(cov_factor)  # Change in the log determinant relative to cov_factor=1
    scaled_inv_cov = inv_cov / cov_factor
    diff_vector = truth - forecast
    scaled_chi_squared = calculate_chi_squared_from_inv_cov(diff_vector, scaled_inv_cov)
    pseudo_log_likeli = -(scaled_chi_squared + scaled_log_det)

    return -pseudo_log_likeli  # Try to minimise neg_log_likeli, will maximise log_likeli


def calculate_chi_squared_from_inv_cov(diff_vector, inv_cov):
    temp_matrix = np.dot(diff_vector.T, inv_cov)
    return np.dot(temp_matrix, diff_vector)


def create_oracle_data_report(oracle_results, output_path):
    """
    Calculate histogram of truth and forecast and save the table to csv
    :param oracle_results: Dataframe indexed by datetime with three columns
    ['returns_actuals', 'returns_forecast_covariance_matrix', 'returns_forecast_mean_vector']
    :param output_path: path where to save the output
    :return: Nope
    """

    oracle_data_table = pd.Series()
    n_samples = len(oracle_results)
    max_return = 0.2
    num_edges = 100
    linear_edges = np.linspace(-max_return, max_return, num_edges)
    log_edges = [1, 10, 100, np.inf]
    neg_log_edges = -np.flip(log_edges, axis=0)

    zeros_in_truth = 0
    zeros_in_forecast = 0
    nans_in_truth = 0
    nans_in_forecast = 0
    infs_in_truth = 0
    infs_in_forecast = 0
    max_zeros_in_single_sample = 0
    max_forecast = 0
    max_truth = 0

    bin_edges = np.concatenate((neg_log_edges, linear_edges, log_edges))

    for i in range(n_samples):
        date = oracle_results.index[i]
        truth = oracle_results.returns_actuals[date].dropna()
        forecast = oracle_results.returns_forecast_mean_vector[date].dropna()

        if i == 0:
            true_hist = np.histogram(truth, bins=bin_edges)[0]
            forecast_hist = np.histogram(forecast, bins=bin_edges)[0]
        else:
            true_hist += np.histogram(truth, bins=bin_edges)[0]
            forecast_hist += np.histogram(forecast, bins=bin_edges)[0]

        sample_zeros = np.sum(truth == 0)
        max_zeros_in_single_sample = np.maximum(sample_zeros, max_zeros_in_single_sample)
        zeros_in_truth += sample_zeros
        zeros_in_forecast += np.sum(forecast == 0)
        infs_in_truth += np.sum(np.isinf(truth))
        infs_in_forecast += np.sum(np.isinf(forecast))

        nans_in_truth += oracle_results.returns_actuals[date].isnull().sum()
        nans_in_forecast += oracle_results.returns_forecast_mean_vector[date].isnull().sum()

        sample_max_forecast = np.max(np.abs(forecast))
        max_forecast = np.maximum(sample_max_forecast, max_forecast)
        sample_max_truth = np.max(np.abs(truth))
        max_truth = np.maximum(sample_max_truth, max_truth)

    oracle_data_table["max truth"] = max_truth
    oracle_data_table["max forecast"] = max_forecast
    oracle_data_table['Zeros in truth'] = zeros_in_truth
    oracle_data_table['Zeros in forecast'] = zeros_in_forecast
    oracle_data_table['Zeros in single sample'] = max_zeros_in_single_sample
    oracle_data_table['NaNs in truth'] = nans_in_truth
    oracle_data_table['NaNs in forecast'] = nans_in_forecast
    oracle_data_table['Infs in truth'] = infs_in_truth
    oracle_data_table['Infs in forecast'] = infs_in_forecast

    oracle_data_table['bin_edges'] = bin_edges
    oracle_data_table['true_hist'] = true_hist
    oracle_data_table['forecast_hist'] = forecast_hist

    oracle_data_table.to_csv(os.path.join(output_path, 'oracle_data_table.csv'))


def calc_masked_reduced_chi2(chi_squared_array):
    """ Computes reduced chi2 while ignoring the presence of NaN or Infs """

    valid_chi2 = np.ma.masked_invalid(chi_squared_array)

    total_chi2 = valid_chi2.sum()
    n_degrees_of_freedom = len(valid_chi2) - 1

    return total_chi2 / n_degrees_of_freedom


def calc_masked_likelihoods(log_likelihood_array):
    """ Computes total and per-sample likelihoods, ignoring the presence of NaN or Infs  """

    valid_likelihoods = np.ma.masked_invalid(log_likelihood_array)
    total_l = valid_likelihoods.sum()
    l_per_sample = total_l / len(valid_likelihoods)

    return total_l, l_per_sample


def calc_masked_forecasts(truth, forecast, covariance, optimal_covariance):
    """ Return only those elements where both truth and forecast are not nans """

    valid_elements = ~np.ma.masked_invalid(truth + forecast).mask

    valid_forecast = forecast[valid_elements]
    valid_truth = truth[valid_elements]
    valid_covariance = covariance[valid_elements[:], :]
    valid_covariance = valid_covariance[:, valid_elements[:]]

    valid_optimal_covariance = optimal_covariance[valid_elements[:], :]
    valid_optimal_covariance = valid_optimal_covariance[:, valid_elements[:]]

    return valid_truth, valid_forecast, valid_covariance, valid_optimal_covariance


def compute_covariance_inverses(oracle_results):
    """Precompute inverses so the optimiser doesnt have to do it each time"""

    n_samples = len(oracle_results)
    inv_covariance_matrices = []

    for i in range(n_samples):
        date = oracle_results.index[i]
        covariance = oracle_results.returns_forecast_covariance_matrix[date]
        covariance = prepare_covariance_for_inversion(covariance)

        inv_covariance_matrices.append(np.linalg.inv(covariance))

    return inv_covariance_matrices


def prepare_covariance_for_inversion(covariance):
    """ Ensure there are no zeros on the diagonal """

    covariance = np.asarray(covariance)
    diag_vals = np.diag(covariance)
    indices = (diag_vals == 0)
    diag_vals.setflags(write=1)
    diag_vals[indices] = 100.0
    np.fill_diagonal(covariance, diag_vals)

    return covariance


def calculate_winners_losers(data):
    """ Returns -1 or +1 in place of whether the element is above or below the median. """

    median = np.median(data)
    winners = np.sign(data - median)
    n_draws = np.sum(winners == 0)

    if n_draws > 0:  # Assign tiebreakers at random
        n_tied_losers = len(data) / 2 - np.sum(winners < 0)

        tiebreaker = np.linspace(1, n_draws, n_draws) - n_tied_losers
        tiebreaker = np.sign(tiebreaker - EPSILON)
        np.random.shuffle(tiebreaker)
        winners[winners == 0] = np.sign(tiebreaker)

    return winners


def _get_all_symbols(oracle_results):
    """
    get all unique tick symbols and return as a set
    :param oracle_results: Dataframe indexed by datetime with three columns
    ['returns_actuals', 'returns_forecast_covariance_matrix', 'returns_forecast_mean_vector']
    :return: returns a set of all unique symbols
    """
    symbols = []
    for row in oracle_results['returns_actuals'].iteritems():
        symbols.extend(list(row[1].index))
    return set(symbols)


def _make_df_dict(oracle_results):
    """
    returns a dict where key value pairs are the tick-symbols and dataframe
    :param oracle_results: Dataframe indexed by datetime with three columns
    ['returns_actuals', 'returns_forecast_covariance_matrix', 'returns_forecast_mean_vector']
    :return: returns a dict where key value pairs are the tick-symbols and dataframe
    """
    symbols = _get_all_symbols(oracle_results)

    dict_of_df = {}

    for symbol in symbols:

        time_stamps = []
        returns_actual = []
        returns_forecast_mean = []
        returns_variance = []

        for i in range(oracle_results.shape[0]):

            if symbol in oracle_results['returns_actuals'][i].index:

                time_stamps.append(oracle_results['returns_actuals'].index[i])
                returns_actual.append(oracle_results['returns_actuals'][i][symbol])
                returns_forecast_mean.append(oracle_results['returns_forecast_mean_vector'][i][symbol])
                covariance_matrix = oracle_results['returns_forecast_covariance_matrix'][i]
                if isinstance(covariance_matrix, (np.ndarray, np.generic)):
                    variance = 1.0
                else:
                    variance = covariance_matrix[symbol][symbol]
                returns_variance.append(variance)

        temp_dict = {'time_stamp': time_stamps, 'returns_actual': returns_actual,
                     "returns_forecast_mean": returns_forecast_mean, "returns_variance": returns_variance}
        df = pd.DataFrame.from_dict(temp_dict)
        df.index = df['time_stamp']
        df.drop(['time_stamp'], axis=1, inplace=True)
        dict_of_df[symbol] = df

    return dict_of_df


def create_time_series_plot(oracle_results, output_path):
    """
    make a time-series plot of the target + prediction with errors
    :param oracle_results: Dataframe indexed by datetime with three columns
    ['returns_actuals', 'returns_forecast_covariance_matrix', 'returns_forecast_mean_vector']
    :param output_path: path where to save the output
    :return:
    """
    dict_of_df = _make_df_dict(oracle_results)
    dicts_for_page = {}
    with PdfPages(os.path.join(output_path, 'time-series-plot.pdf')) as pdf:
        for pid, (symbol, df) in enumerate(dict_of_df.items()):
            dicts_for_page[symbol] = df

            if len(dicts_for_page) >= 2 or pid == len(dict_of_df) - 1:
                fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(11.69, 8.27), sharex=False)
                for i, (symb_page, df_page) in enumerate(dicts_for_page.items()):
                    df_page.index = pd.to_datetime(df_page.index, format="%Y%m%d-%H%M%S")
                    df_page.index = df_page.index.format(formatter=lambda x: x.strftime('%Y%m%d'))
                    axes[i].scatter(df_page.index, df_page.returns_actual, color='navy')
                    axes[i].plot(df_page.index, df_page.returns_actual, color='navy', label='')

                    axes[i].scatter(df_page.index, df_page.returns_forecast_mean, color='firebrick')
                    axes[i].plot(df_page.index, df_page.returns_forecast_mean, color='firebrick', label='')

                    axes[i].fill_between(df_page.index,
                                         df_page.returns_forecast_mean - np.sqrt(df_page.returns_variance),
                                         df_page.returns_forecast_mean + np.sqrt(df_page.returns_variance),
                                         color='firebrick', alpha=0.3)
                    axes[i].legend(loc=1, prop={'size': 10})
                    axes[i].set_title(symb_page)
                    axes[i].set_ylabel('Log Returns')
                    axes[i].tick_params(axis='x', labelsize=10)
                    axes[i].tick_params(axis='y', labelsize=10)
                    axes[i].xaxis.set_major_locator(MaxNLocator(DEFAULT_MAX_N_TICKS))
                    [lc.set_rotation(15) for lc in axes[i].get_xticklabels()]
                    [lc.set_rotation(15) for lc in axes[i].get_yticklabels()]

                if len(dicts_for_page) == 1:
                    fig.delaxes(axes[1])

                pdf.savefig(fig)
                plt.close()

                dicts_for_page = {}


def calculate_weighted_correlation_coefficient(truth, forecast, oracle_symbol_weights):
    """ Estimate the correlation coefficient with weights

    :param truth:
    :param forecast:
    :param oracle_symbol_weights:
    :return:
    """

    if oracle_symbol_weights is None:
        oracle_symbol_weights = np.ones(shape=truth.shape)

    # Filter out nans
    indices = ~np.isnan(truth + forecast)
    truth = truth[indices]
    forecast = forecast[indices]
    oracle_symbol_weights = oracle_symbol_weights[indices]

    def cov(x, y, w):
        """Weighted Covariance"""
        m = np.average(x, weights=w)
        result = np.sum(w * (x - m) * (y - m)) / np.sum(w)
        return result

    def corr(x, y, w):
        """Weighted Correlation"""
        return cov(x, y, w) / np.sqrt(cov(x, x, w) * cov(y, y, w))

    if len(truth) > 0:
        correlation = corr(truth, forecast, oracle_symbol_weights)
    else:
        correlation = np.nan

    return correlation


def extract_weight_array(predicted_symbols, oracle_symbol_weights):
    """

    :param list of predicted_symbols:
    :param pdSeries oracle_symbol_weights:
    :return: nparray weights of the predicted_symbols
    """

    n_predicted_symbols = len(predicted_symbols)
    weights = np.zeros(shape=n_predicted_symbols)

    for i in range(n_predicted_symbols):
        symbol = predicted_symbols[i]
        weights[i] = oracle_symbol_weights.loc[symbol]

    return weights


def create_moving_average_figure(time_series, series_name, rolling_window):
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(111)

    y_axis_formatter = FuncFormatter(lambda x, pos: '%.2f' % x)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))
    ax.set_title('{}'.format(series_name))
    ax.set_ylabel('{}'.format(series_name))
    ax.set_xlabel('')

    rolling_ts = time_series.rolling(rolling_window).mean()

    time_series.plot(alpha=.3, lw=3, color='blue', ax=ax)
    rolling_ts.plot(alpha=.7, lw=3, color='orangered', ax=ax)

    ax.axhline(
        time_series.mean(),
        color='steelblue',
        linestyle='--',
        lw=3)
    ax.axhline(0.0, color='black', linestyle='-', lw=3)

    ax.legend(['Spot', 'Moving Average ({} points)'.format(rolling_window), 'Mean'],
              loc='best', frameon=True, framealpha=0.5)

    plt.setp(ax.get_xticklabels(), visible=True)

    return fig


def create_time_series_comparison_figure(time_series_list, series_name_list, y_label):
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(111)

    y_axis_formatter = FuncFormatter(lambda x, pos: '%.2f' % x)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))
    ax.set_title('{}'.format(y_label))
    ax.set_ylabel('{}'.format(y_label))
    ax.set_xlabel('')

    for time_series in time_series_list:
        time_series.plot(lw=3, ax=ax)

    ax.legend(series_name_list, loc='best', frameon=True, framealpha=0.5)

    plt.setp(ax.get_xticklabels(), visible=True)

    return fig


def get_results_file(path, ends_with, required=True, starts_with=None):
    all_files_in_path = os.listdir(path)
    matching_file_list = []

    for file in all_files_in_path:
        if starts_with:
            if file.endswith(ends_with) and file.startswith(starts_with):
                matching_file_list.append(file)
        else:
            if file.endswith(ends_with):
                matching_file_list.append(file)

    if len(matching_file_list) == 0 and not required:
        return None

    assert len(matching_file_list) == 1, \
        'Found {} files matching the suffix {}: it must be one!'.format(len(matching_file_list), ends_with)

    return os.path.join(path, matching_file_list[0])


def read_oracle_results_from_path(results_path, run_mode=None):
    oracle_results_mean_vector_filepath = get_results_file(results_path, ORACLE_MEAN_VECTOR_ENDSWITH_TEMPLATE,
                                                           starts_with=run_mode)
    oracle_results_covariance_matrix_filepath = get_results_file(results_path,
                                                                 ORACLE_COVARIANCE_MATRIX_ENDSWITH_TEMPLATE,
                                                                 starts_with=run_mode)
    oracle_results_actuals_filepath = get_results_file(results_path, ORACLE_ACTUALS_ENDSWITH_TEMPLATE,
                                                       starts_with=run_mode)

    oracle_results = read_oracle_results_files(oracle_results_mean_vector_filepath,
                                               oracle_results_covariance_matrix_filepath,
                                               oracle_results_actuals_filepath)
    return oracle_results


def read_oracle_results_files(mean_vector_file, covariance_matrix_file, actuals_file):
    store_mean_vector = pd.HDFStore(mean_vector_file)
    store_covariance_matrix = pd.HDFStore(covariance_matrix_file)
    store_actuals = pd.HDFStore(actuals_file)

    timestamps = extract_matching_timestamps(store_mean_vector, store_covariance_matrix, store_actuals)
    oracle_results = pd.DataFrame(columns=ORACLE_METRIC_COLUMNS)

    for dt in timestamps:
        tmp_df = pd.DataFrame([[store_mean_vector.get(dt), store_covariance_matrix.get(dt), store_actuals.get(dt)]],
                              index=[dt], columns=ORACLE_METRIC_COLUMNS)
        oracle_results = oracle_results.append(tmp_df)

    store_mean_vector.close()
    store_covariance_matrix.close()
    store_actuals.close()

    return oracle_results


def read_oracle_symbol_weights_from_path(results_path):
    oracle_symbol_weights_filepath = get_results_file(results_path, ORACLE_SYMBOL_WEIGHTS_ENDSWITH_TEMPLATE,
                                                      required=False)
    if oracle_symbol_weights_filepath is None:
        oracle_symbol_weights = None
    else:
        oracle_symbol_weights = pd.Series.from_csv(oracle_symbol_weights_filepath)

    return oracle_symbol_weights


def extract_matching_timestamps(df, reference_a, reference_b):
    """ Collects timestamps from first argument which also appear in other two

    :param df:
    :param reference_a:
    :param reference_b:
    :return:
    """
    mean_timestamps = [table[1:] for table in df]
    reference_a_timestamps = [table[1:] for table in reference_a]
    reference_b_timestamps = [table[1:] for table in reference_b]

    matching_timestamps = []
    for timestamp in mean_timestamps:
        if timestamp in reference_a_timestamps and timestamp in reference_b_timestamps:
            matching_timestamps.append(timestamp)
        else:
            logger.warning('Incomplete prediction. The following timestamp will be ignored: {}'.format(str(timestamp)))

    return matching_timestamps


def check_validity_of_covariances(oracle_results):
    valid_covariances = True

    n_samples = len(oracle_results)
    for i in range(n_samples):
        date = oracle_results.index[i]
        covariance = oracle_results.returns_forecast_covariance_matrix[date]

        forecast = oracle_results.returns_forecast_mean_vector[date].dropna().values

        n_predict = len(forecast)
        cov_size = covariance.shape

        if n_predict != cov_size[0] or n_predict != cov_size[1]:
            valid_covariances = False
            logger.info("Invalid covariance matrix found:"
                        "shape {} c.f. prediction length {}".format(cov_size, n_predict))
            break

    return valid_covariances
