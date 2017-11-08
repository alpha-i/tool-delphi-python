from abc import ABCMeta, abstractmethod


class AbstractDataSource(metaclass=ABCMeta):

    def __init__(self):
        self.name = "name"
        self.train_data = None
        self.test_data = None

    @abstractmethod
    def get_sample(self, query, prediction_event):
        raise NotImplementedError()

    @abstractmethod
    def get_train_data(self):
        raise NotImplementedError()

    @abstractmethod
    def get_test_data(self):
        raise NotImplementedError()

    @abstractmethod
    def get_actual(self, prediction_event):
        raise NotImplementedError()



