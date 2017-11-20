from abc import ABCMeta, abstractmethod


class AbstractDataSource(metaclass=ABCMeta):

    def __init__(self, config):
        self._train_start = config.train_start
        self._train_end = config.train_end
        self._test_start = config.test_start
        self._test_end = config.test_end

    @abstractmethod
    def get_dev_data(self):
        raise NotImplementedError()

    @abstractmethod
    def get_validation_data(self):
        raise NotImplementedError()

    @abstractmethod
    def get_test_data(self):
        raise NotImplementedError()

    @abstractmethod
    def get_data_window(self, data, start, end):

        data_window = {}

        for key in data:
            data_window[key] = data[key][start, end]

        return data_window


