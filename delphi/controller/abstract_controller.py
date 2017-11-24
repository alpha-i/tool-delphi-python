from abc import ABCMeta, abstractmethod

from delphi.oracle.performance import OraclePerformance


class AbstractController(metaclass=ABCMeta):
    def __init__(self, configuration, oracle, datasource, scheduler, performance):
        """
        :param configuration:
        :type configuration: ControllerConfiguration
        :param oracle: The oracle to run
        :type oracle: AbstractOracle
        :param datasource: The source of the data
        :type datasource: AbstractDataSource
        :param scheduler: the Scheduler
        :type scheduler: AbstractScheduler
        :param performance: the performance class
        :type performance: OraclePerformance
        """
        self.configuration = configuration
        self.oracle = oracle
        self.datasource = datasource
        self.scheduler = scheduler
        self.performance = performance
        self.prediction_results = []

    @abstractmethod
    def run(self):
        raise NotImplementedError
