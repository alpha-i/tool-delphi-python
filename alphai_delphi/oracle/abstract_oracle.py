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
        :param timestamp: timestamp of the predicted date
        :type timestamp: datetime
        """
        self.covariance_matrix = covariance_matrix
        self.mean_vector = mean_vector
        self.timestamp = timestamp

    def __repr__(self):
        return "<Prediction result: {}>".format(self.__dict__)


class AbstractOracle(metaclass=ABCMeta):
    def __init__(self, config):
        """
        :param config:
        :type config: OracleConfiguration
        """
        self.scheduling = config.scheduling
        self.config = config.oracle

    @abstractmethod
    def save(self):
        """
        Save the trained oracle state
        """
        raise NotImplementedError

    @abstractmethod
    def load(self):
        """
        Method to load a state for the ML model
        """
        raise NotImplementedError

    @abstractmethod
    def train(self, data, current_timestamp):
        """
        Main method for training our ML model

        :param data: The dict of dataframes to be used for training
        :type data: dict:
        :param current_timestamp: The timestamp of the time when the train is executed
        :type current_timestamp: datetime
        :return: void
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, data, current_timestamp, target_timestamp):
        """
        Main method that gives us a prediction after the training phase is done

        :param data: The dict of dataframes to be used for prediction
        :type data: dict
        :param current_timestamp: The timestamp of the time when the prediction is executed
        :type current_timestamp: datetime.datetime
        :param target_timestamp: The timestamp of the point in time we are predicting
        :type target_timestamp: datetime.datetime
        :return: Mean vector or covariance matrix together with the timestamp of the prediction
        :rtype: PredictionResult
        """
        raise NotImplementedError

    def _preprocess_raw_data(self, data):
        data = self.fill_nan(data)
        data = self.resample(data)
        return self.global_transform(data)

    @abstractmethod
    def resample(self, data):
        raise NotImplementedError

    @abstractmethod
    def fill_nan(self, data):
        raise NotImplementedError

    @abstractmethod
    def global_transform(self, data):
        """
        does resampling and  global transformations

        :param data:  The dict of dataframes
        :type data: dict
        :return: dict of dataframes
        """
        raise NotImplementedError

    @abstractmethod
    def get_universe(self, data):
        """
        returns universe based on transformed data

        :param data:  The dict of dataframes
        :type data: dict
        :return: dataframe
        """
        raise NotImplementedError

    @property
    def training_frequency(self):
        """
        Frequency upon which we do a training

        :rtype: SchedulingFrequency
        """
        return self.scheduling.training_frequency

    @property
    def prediction_frequency(self):
        """
        Frequency upon which we do a prediction

        :rtype: SchedulingFrequency
        """
        return self.scheduling.prediction_frequency

    @property
    def prediction_horizon(self):
        """
        :rtype: datetime.timedelta
        """
        return self.scheduling.prediction_horizon

    @property
    def prediction_offset(self):
        """
        Amount of time from the market open
        :rtype: datetime.timedelta
        """
        return self.scheduling.prediction_offset

    @property
    @abstractmethod
    def target_feature(self):
        """
        This is the feature name that will be predicted (Must be present in the input data too)
        :rtype: str
        """
        raise NotImplementedError

    def get_delta_for_event(self, event):
        """
        Given a schedule event, returns the interval of data
        that the oracle wants to be passed to it by the data_source
        :param event:
        :type
        :return: interval
        :rtype: datetime.timedelta
        """
        if event == OracleAction.PREDICT:
            return self.scheduling.prediction_delta
        elif event == OracleAction.TRAIN:
            return self.scheduling.training_delta

        raise ValueError("Event type {} not supported".format(event))
