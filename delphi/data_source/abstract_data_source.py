from abc import ABCMeta, abstractmethod


class AbstractDataSource(metaclass=ABCMeta):
    def __init__(self, configuration):
        self.configuration = configuration

    @property
    @abstractmethod
    def current_minute(self):
        raise NotImplemented

    @abstractmethod
    def get_data(self, interval):
        """
        Yeah, get data

        :param interval: The interval before the current minute to get data for
        :type interval: datetime.timedelta
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
