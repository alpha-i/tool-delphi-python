from abc import ABCMeta, abstractmethod


class AbstractDataProvider(metaclass=ABCMeta):

    @abstractmethod
    def get_data(self, start_date, end_date):
        pass
