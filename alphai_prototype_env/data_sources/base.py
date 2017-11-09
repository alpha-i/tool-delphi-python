from abc import ABCMeta, abstractmethod


class AbstractDataSource(metaclass=ABCMeta):

    @abstractmethod
    def get_train_data(self):
        raise NotImplementedError()

    @abstractmethod
    def get_test_data(self):
        raise NotImplementedError()
