import logging
from abc import ABCMeta, abstractmethod

logging.getLogger(__name__).addHandler(logging.NullHandler())


class AbstractController(metaclass=ABCMeta):
    def __init__(self, configuration, oracle, datasource, scheduler):
        """
        :param configuration:
        :type configuration: dict
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

    @abstractmethod
    def run(self):
        raise NotImplementedError


