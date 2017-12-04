from abc import ABCMeta, abstractmethod


class AbstractDataSource(metaclass=ABCMeta):
    def __init__(self, configuration):
        """
        A blueprint for possible data sources

        :param configuration:
        :type configuration: dict
        """
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

    @abstractmethod
    def values_for_symbols_feature_and_time(self, symbol_list, feature, current_datetime):
        """
        Return the value for in the coordinates identified by feature, datetime and list of symbol.
        The current data has the following structure.
        {
            feature =>           | symbol1 | symbol2 | symbol3 |
                        12:00:00 |   1.3   |   0.5   |  0.3    |
                        12:01:00 |   1.3   |   0.5   |  0.3    |
                        12:02:00 |   1.3   |   0.5   |  0.3    |
                        12:03:00 |   1.3   |   0.5   |  0.3    |
                        12:04:00 |   1.3   |   0.5   |  0.3    |
        }
        :param symbol_list: the list of symbols we want to get value for
        :type symbol_list: list
        :param str feature:
        :type feature: str
        :param current_datetime:
        :type current_datetime: datetime.datetime
        :return:
        :rtype: np.array
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
