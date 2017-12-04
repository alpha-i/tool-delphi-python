import logging

import numpy as np

from alphai_delphi.controller.abstract_controller import AbstractController
from alphai_delphi.controller.controller_configuration import ControllerConfiguration
from alphai_delphi.oracle import OracleAction
from alphai_delphi.scheduler import ScheduleException


class Controller(AbstractController):

    def run(self):
        for moment, events in self.scheduler:
            for action in events:

                interval = self.oracle.get_delta_for_event(action)
                raw_data = self.datasource.get_data(moment, interval)

                if action == OracleAction.TRAIN:
                    logging.debug("Training at {}".format(moment))
                    self.oracle.train(raw_data, moment)
                elif action == OracleAction.PREDICT:
                    self._record_actual_performance(self.oracle.target_feature, moment)
                    try:
                        target_moment = self.scheduler.get_first_valid_target(moment, self.oracle.prediction_horizon)
                    except ScheduleException as e:
                        logging.debug(e)
                        continue
                    logging.debug("Prediction at {}".format(moment))
                    prediction_result = self.oracle.predict(raw_data, moment, target_moment)
                    self.prediction_results.append(prediction_result)
                    self._record_prediction(moment, self.oracle.target_feature, prediction_result)

    def _record_actual_performance(self, feature_name, current_dt):
        """

        :param feature_name:
        :type feature_name: str
        :param current_dt:
        :type current_dt: datetime.datetime
        """
        if current_dt in self.performance:
            previous_symbols = self.performance.get_symbols(current_dt)
            final_values = self.datasource.values_for_symbols_feature_and_time(
                previous_symbols,
                feature_name,
                current_dt
            )
            self.performance.add_final_values(current_dt, final_values)
            self.performance.save_to_hdf5(current_dt)
            self.performance.drop_dt(current_dt)

    def _record_prediction(self, current_datetime, feature_name, prediction_result):
        target_dt = prediction_result.timestamp
        prediction_symbols = np.array(prediction_result.mean_vector.index)
        initial_values = self.datasource.values_for_symbols_feature_and_time(
            prediction_symbols,
            feature_name,
            current_datetime
        )
        self.performance.add_prediction(target_dt, prediction_result.mean_vector, prediction_result.covariance_matrix)
        self.performance.add_initial_prices(target_dt, initial_values)
