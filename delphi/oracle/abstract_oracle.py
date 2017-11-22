from abc import ABCMeta, abstractmethod
from enum import Enum


class OracleAction(Enum):
    PREDICT = 0
    TRAIN = 1


class PredictionResult:
    def __init__(self, mean_vector, covariance_matrix, timestamp):
        """

        :param mean_vector: vector of predicted means
        :type mean_vector: pd.Series
        :param covariance_matrix: covariance matrix
        :type covariance_matrix: pd.DataFrame
        :param timestamp: timestamp of the prediction
        :type timestamp: datetime
        """
        self.covariance_matrix = covariance_matrix
        self.mean_vector = mean_vector
        self.timestamp = timestamp


class AbstractOracle(metaclass=ABCMeta):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def save(self):
        """
        Save the trained oracle state
        """
        raise NotImplemented

    @abstractmethod
    def load(self):
        """
        Method to load a state for the ML model
        """
        raise NotImplemented

    @abstractmethod
    def train(self, data):
        """
        Main method for training our ML model

        :return: void
        """
        raise NotImplemented

    @abstractmethod
    def predict(self, data, timestamp):
        """
        Main method that gives us a prediction after the training phase is done

        :param data: The dict of dataframes to be used for prediction
        :type data: dict
        :param timestamp: The timestamp of the point in time we are predicting
        :type timestamp: datetime
        :return: Mean vector or covariance matrix together with the timestamp of the prediction
        :rtype: PredictionResult
        """
        raise NotImplemented

    @property
    def train_frequency(self):
        """
        Frequency upon which we do a training

        :rtype: SchedulingFrequency
        """
        return self.config["train_frequency"]

    @property
    def predict_frequency(self):
        """
        Frequency upon which we do a prediction

        :rtype: SchedulingFrequency
        """
        return self.config["predict_frequency"]

    @property
    def predict_horizon(self):
        """
        :rtype: datetime.timedelta
        """
        return self.config["predict_horizon"]

    @property
    def predict_offset(self):
        """
        Amount of time from the market open
        :rtype: datetime.timedelta
        """
        return self.config["predict_offset"]

    @abstractmethod
    def get_interval(self, event):
        """
        Given a schedule event, returns the interval of data
        that the oracle wants to be passed to it by the data_source
        :param event:
        :type
        :return: interval
        :rtype: datetime.timedelta
        """

        raise NotImplementedError()
