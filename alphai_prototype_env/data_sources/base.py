from abc import ABCMeta, abstractmethod


class AbstractDataSource(metaclass=ABCMeta):

    def __init__(self, config):
        self._train_start = config.train_start
        self._train_end = config.train_end
        self._validation_start = config.validation_start
        self._validation_end = config.validation_end
        self._test_start = config.test_start
        self._test_end = config.test_end

    @abstractmethod
    def get_data(self, mode):

        if mode == "develop":
            data = self.get_dev_data()
        elif mode == "validation":
            data = self.get_validation_data()
        elif mode == "test":
            data = self.get_test_data()
        else:
            raise ValueError('Unexpected mode: {}'.format(mode))

        return data

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

    @abstractmethod
    def get_start_end_datetimes(self, mode):

        if mode == "train":
            start = self._train_start
            end = self._train_end
        elif mode == "validation":
            start = self._validation_start
            end = self._validation_end
        elif mode == "test":
            start = self._test_start
            end = self._test_end
        else:
            raise ValueError('Unexpected mode: {}'.format(mode))

        return start, end


