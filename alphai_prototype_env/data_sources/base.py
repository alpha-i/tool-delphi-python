from abc import ABCMeta, abstractmethod


class AbstractDataSource(metaclass=ABCMeta):

    def __init__(self, config):
        self.feature_vars = config.feature_vars
        self.prediction_vars = config.prediction_vars
        self._train_start = config.train_start
        self._train_end = config.train_end
        self._test_start = config.test_start
        self._test_end = config.test_end

    @abstractmethod
    def get_train_data(self):
        raise NotImplementedError()

    @abstractmethod
    def get_test_data(self):
        raise NotImplementedError()
