from abc import ABCMeta, abstractmethod


class AbstractDataSource(metaclass=ABCMeta):
    def __init__(self, configuration):
        self.config = configuration

    @abstractmethod
    def get_data(self, current_datetime, interval):
        """
        :param current_datetime: the current date and time to get data up until
        :type current_datetime: datetime.datetime
        :param interval: the interval of time to look back into the past
        :type timedelta: datetime.timedelta
        :return: a dictionary of data frames
        :rtype dict
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def start(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def end(self):
        raise NotImplementedError
