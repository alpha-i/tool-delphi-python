from abc import ABCMeta, abstractmethod

from alphai_delphi.configuration.schemas import ControllerConfigurationSchema, InvalidConfiguration
from alphai_delphi.performance.performance import OraclePerformance


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
        data, errors = ControllerConfigurationSchema().load(configuration)
        if errors:
            raise InvalidConfiguration(errors)
        self.configuration = data
        self.oracle = oracle
        self.datasource = datasource
        self.scheduler = scheduler
        self.performance = performance
        self.prediction_results = []

        self.name = self.performance.run_mode
        self.start_time = None
        self.end_time = None
        self.simulation_start = self.scheduler.start_date
        self.simulation_end = self.scheduler.end_date
        self.prediction_moments = []

    @abstractmethod
    def run(self):
        raise NotImplementedError
