import datetime
from abc import ABCMeta, abstractmethod
from enum import Enum


class SchedulingFrequencyType(Enum):
    DAILY = 0
    MINUTE = 1
    WEEKLY = 2
    NEVER = 3


class AbstractScheduler(metaclass=ABCMeta):
    def __init__(self, start_date, end_date,
                 prediction_frequency, training_frequency):
        """
        :param start_date: The beginning of the full scheduling window
        :type start_date: datetime.datetime
        :param end_date: The end of the scheduling window
        :type end_date: datetime.datetime
        :param training_frequency: Defines the frequency of training to build a schedule
        :type training_frequency: SchedulingFrequency
        :param prediction_frequency: Define the frequency of Prediction to build a schedule
        :type prediction_frequency: SchedulingFrequency

        """
        self.start_date = start_date
        self.end_date = end_date
        self.prediction_frequency = prediction_frequency
        self.training_frequency = training_frequency

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError

    @abstractmethod
    def get_event(self, minute):
        """
        Given a minute, give back the list of action(s) that the oracle is supposed to perform

        :param minute: Minute in time to get an event for
        :type minute: datetime.datetime
        :return: List of actions to be performed by the oracle
        :rtype: List[OracleActions]
        """
        raise NotImplementedError

    @abstractmethod
    def get_first_valid_target(self, moment, interval):
        """
        Given a datetime and an interval, give back the first suitable datetime according to market schedule

        :param moment:
        :type moment: datetime.datetime
        :param interval:
        :type interval: datetime.timedelta
        :return: The first suitable datetime
        :rtype: datetime.datetime
        """
        raise NotImplementedError


class ScheduleException(Exception):
    pass
