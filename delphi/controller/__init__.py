import logging

from delphi.controller.abstract_controller import AbstractController
from delphi.controller.controller_configuration import ControllerConfiguration
from delphi.oracle import OracleAction
from delphi.scheduler import ScheduleException


class Controller(AbstractController):

    def __init__(self, configuration, oracle, datasource, scheduler):
        super().__init__(configuration, oracle, datasource, scheduler)

    def run(self):
        for moment, events in self.scheduler:
            for action in events:
                interval = self.oracle.get_delta_for_event(action)
                data = self.datasource.get_data(moment, interval)

                if action == OracleAction.TRAIN:
                    logging.debug("Training at {}".format(moment))
                    self.oracle.train(data)
                elif action == OracleAction.PREDICT:
                    try:
                        target_moment = self.scheduler.get_first_valid_target(moment, self.oracle.prediction_horizon)
                    except ScheduleException as e:
                        logging.debug(e)
                        continue
                    logging.debug("Prediction at {}".format(moment))
                    prediction_result = self.oracle.predict(data, target_moment)
