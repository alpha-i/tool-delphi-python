from abc import ABCMeta, abstractmethod

import pandas as pd


class AbstractOracle(metaclass=ABCMeta):

    def __init__(self, config):
        # Define the schedule on which trading will occur and the inputs and predicted outputs

        # (for now will assume that models can only make a prediction at a single point in time)
        self.trade_frequency = config.trade_frequency  # in minutes
        self.trade_delta = config.trade_delta  # in minutes
        self.trade_offset = config.trade_offset  # in minutes
        self.past_horizon = config.past_horizon  # time in minutes to look back for
        self.retrain_frequency = config.retrain_frequency

    @abstractmethod
    def reset(self ):
        raise NotImplementedError()

    @abstractmethod
    def save(self, filepath):
        """
        Save model and parameters to a file
        :param filepath: string
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def load(self, filepath):
        """
        Load an existing model from a filepath
        :param filepath: string
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def transform(self, data):
        """
        Transform the OHLCV dict of dataframes into whatever output is required by the ML model
        :param data: OHLCV data as dictionary of pandas DataFrames.
        :return: transformed_data
        """
        raise NotImplementedError()

    @abstractmethod
    def train(self, data):
        """
        :param train_data: OHLCV data as dictionary of pandas DataFrames.
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def predict(self, data):
        """
        :param predict_data: OHLCV data as dictionary of pandas DataFrames.
        :return:
        """
        prediction = (pd.Series(), pd.DataFrame)

        return prediction
