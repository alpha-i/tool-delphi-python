from abc import ABCMeta, abstractmethod


class AbstractDataSource(metaclass=ABCMeta):
    def __init__(self, configuration):
        self.config = configuration

    @abstractmethod
    def get_data(self, current_datetime, interval):
        """
        :param current_datetime, datetime
        :param interval: timedelta
        :type data_dict: dict
        :return:
        """
        raise NotImplemented

    @property
    @abstractmethod
    def start(self):
        raise NotImplemented

    @property
    @abstractmethod
    def end(self):
        raise NotImplemented
