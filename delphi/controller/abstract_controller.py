import logging
from abc import ABCMeta, abstractmethod

from delphi.oracle.performance import OraclePerformance

logging.getLogger(__name__).addHandler(logging.NullHandler())


class AbstractController(metaclass=ABCMeta):
    def __init__(self, configuration, oracle, datasource, scheduler):
        """
        :param configuration:
        :type configuration: ControllerConfiguration
        :param oracle: The oracle to run
        :type oracle: AbstractOracle
        :param datasource: The source of the data
        :type datasource: AbstractDataSource
        :param scheduler: the Scheduler
        :type scheduler: AbstractScheduler
        """
        self.configuration = configuration
        self.oracle = oracle
        self.datasource = datasource
        self.scheduler = scheduler
        self.performance = OraclePerformance(
            configuration.performance_result_output,
            'delphi'
        )

    @abstractmethod
    def run(self):
        raise NotImplementedError


