from abc import ABCMeta, abstractmethod
from enum import Enum


class OracleActions(Enum):
    PREDICT = 0
    TRAIN = 1


class PredictionResult:
    def __init__(self, mean_vector, covariance_matrix, timestamp):
        self.covariance_matrix = covariance_matrix
        self.mean_vector = mean_vector
        self.timestamp = timestamp


class AbstractOracle(metaclass=ABCMeta):
    def __init__(self, configuration):
        self.configuration = configuration

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
    def train(self, datasource):
        """
        Main method for training our ML model

        :return: void
        """
        raise NotImplemented

    @abstractmethod
    def predict(self, datasource):
        """
        Main method that gives us a prediction after the training phase is done

        :param datasource: The data source used for prediction
        :type datasource: DataSource
        :return: Minute vector or covariance method together with the timestamp of the prediction
        :rtype: PredictionResult
        """
        raise NotImplemented

    @property
    @abstractmethod
    def train_frequency(self):
        """
        Frequency upon which we do a training

        :rtype: SchedulingFrequency
        """
        raise NotImplemented

    @property
    @abstractmethod
    def predict_frequency(self):
        """
        Frequency upon which we do a prediction

        :rtype: SchedulingFrequency
        """
        raise NotImplemented

    @property
    @abstractmethod
    def predict_horizon(self):
        """
        :rtype: datetime.timedelta
        """
        raise NotImplemented

    @property
    @abstractmethod
    def predict_offset(self):
        """
        Amount of time from the market open
        :rtype: datetime.timedelta
        """
        raise NotImplemented
