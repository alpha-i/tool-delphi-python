from abc import ABCMeta, abstractmethod


class AbstractMetrics(metaclass=ABCMeta):
    @abstractmethod
    def compute_metrics(self, predictions, actuals):
        """

        :param predictions:
        :
        :param actuals:
        :return:
        """
        raise NotImplementedError
