from abc import ABCMeta, abstractmethod


class AbstractDataSource(metaclass=ABCMeta):
    def __init__(self, configuration):
        self.config = configuration

    @abstractmethod
    def get_data(self, current_datetime, interval):
        """
        Yeah, get data

        :param current_datetime, datetime
        :param interval: The interval before the current minute to get data for
        :type data_dict: dict of data frames
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
