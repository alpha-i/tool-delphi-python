from abc import ABCMeta, abstractmethod


class AbstractOracle(metaclass=ABCMeta):

    def __init__(self, frequency, delta, offset, past_horizon, feature_vars, prediction_vars):
        # Define the schedule on which trading will occur and the inputs and predicted outputs

        # (for now will assume that models can only make a prediction at a single point in time)
        self.frequency = frequency  # in minutes
        self.delta = delta  # in minutes
        self.offset = offset  # in minutes
        self.past_horizon = past_horizon  # time in minutes to look back for
        self.feature_vars = feature_vars  # e.g., {"close": ['AAPL', 'MSFT'], "volume": ['AAPL', 'MSFT']}
        self.prediction_vars = prediction_vars  # e.g., {"close": ['AAPL']}

    @abstractmethod
    def transform(self, data):
        """
        Transform the OHLCV dict of dataframes into whatever output is required by the ML model
        :param data: OHLCV data as dictionary of pandas DataFrames.
        :return: transformed_data
        """
        raise NotImplementedError()

    @abstractmethod
    def train(self, train_data):
        """
        :param train_data: OHLCV data as dictionary of pandas DataFrames.
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def predict(self, predict_data):
        """
        :param predict_data: OHLCV data as dictionary of pandas DataFrames.
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
    def save(self, filepath):
        """
        Save model and parameters to a file
        :param filepath: string
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def get_batch(self, transformed_train_data):
        """
        Get a random batch of training data of size self.batch_size
        :param transformed_train_data:
        :return: features, labels
        """
        raise NotImplementedError()
