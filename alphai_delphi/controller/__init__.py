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
                    self._do_train(raw_data, moment)
                elif action == OracleAction.PREDICT:
                    try:
                        target_moment = self.scheduler.get_first_valid_target(moment, self.oracle.prediction_horizon)
                    except ScheduleException as e:
                        logging.debug(e)
                        continue
                    self._do_predict(raw_data, moment, target_moment)

        self.performance.create_oracle_report()

    def _do_train(self, raw_data, current_moment):
        """
        Perfrorm Training

        :param raw_data:
        :type raw_data: dict
        :param current_moment:
        :type current_moment: datetime.datetime
        """
        logging.info("START training at {}".format(current_moment))
        try:
            self.oracle.train(raw_data, current_moment)
            logging.info("END training at {}".format(current_moment))
        except ValueError as e:
            logging.error("SKIP training. Reason: {}".format(e))

    def _do_predict(self, raw_data, current_moment, target_moment):
        """
        Performs prediction

        :param raw_data: dict of pd.DataFrame
        :type raw_data: dict
        :param current_moment:
        :type current_moment: datetime.datetime
        :param target_moment:
        :type datetime.datetime
        :return:
        """
        self._record_actual_performance(self.oracle.target_feature, current_moment)
        logging.info("START prediction at {}".format(current_moment))
        try:
            prediction_result = self.oracle.predict(raw_data, current_moment, target_moment)
            self.prediction_results.append(prediction_result)
            self._record_prediction(current_moment, self.oracle.target_feature, prediction_result)
            logging.info("END prediction at {}".format(current_moment))
        except ValueError as e:
            logging.error("SKIP prediction. Reason {}".format(e))



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
