from abc import ABCMeta, abstractmethod

from delphi.oracle import OracleAction
from delphi.scheduler.abstract_scheduler import ScheduleException


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


class Controller(AbstractController):

    def __init__(self, configuration, oracle, datasource, scheduler):
        super().__init__(configuration, oracle, datasource, scheduler)

    def run(self):
        for moment, events in self.scheduler:
            for action in events:
                interval = self.oracle.get_delta_for_event(action)
                data = self.datasource.get_data(moment, interval)

                if action == OracleAction.TRAIN:
                    print("Training at {}".format(moment))
                    self.oracle.train(data)
                elif action == OracleAction.PREDICT:
                    try:
                        target_moment = self.scheduler.get_first_valid_target(moment, self.oracle.prediction_horizon)
                    except ScheduleException as e:
                        continue
                    print("Prediction at {}".format(moment))
                    prediction_result = self.oracle.predict(data, target_moment)
