import logging
from datetime import datetime
from datetime import timedelta

import numpy as np

from alphai_delphi.controller.abstract_controller import AbstractController
from alphai_delphi.oracle.abstract_oracle import OracleAction
from alphai_delphi.scheduler.scheduler import ScheduleException

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class Controller(AbstractController):
    def __init__(self, configuration, oracle, datasource, scheduler, performance):
        super().__init__(configuration, oracle, datasource, scheduler, performance)
        self.elapsed_time = None
        self.start_time = None

    def run(self):
        self.start_time = datetime.now()
        logger.info("%s run started at %s", self.name, self.start_time)
        for prediction_moment, events in self.scheduler:
            for action in events:
                oracle_interval = self.oracle.get_delta_for_event(action)
                interval = self.get_market_interval(prediction_moment, oracle_interval)
                raw_data = self.datasource.get_data(prediction_moment, interval)

                if action == OracleAction.TRAIN:
                    self._do_train(raw_data, prediction_moment)

                elif action == OracleAction.PREDICT:
                    try:
                        target_moment = self.scheduler.get_first_valid_target(prediction_moment,
                                                                              self.oracle.prediction_horizon)
                    except ScheduleException as e:
                        logger.debug(e)
                        continue
                    prediction_result = self._do_predict(raw_data, prediction_moment, target_moment)
                    if prediction_result:
                        self.prediction_moments.append(
                            (prediction_moment, prediction_result.prediction_timestamp,
                             prediction_result.target_timestamp)
                        )
        self.end_time = datetime.now()
        self.elapsed_time = self.end_time - self.start_time
        logger.info("%s finished at %s. Took %s", self.name, self.end_time, self.elapsed_time)

        self.performance.create_oracle_report()
        self.print_run_summary()

    def get_market_interval(self, moment, oracle_interval):
        """
        Given a moment and an interval from the oracle, it calculates a correct business day intervall
        It always add 1 day more to the interval to prevent any missing data on the datasource.
        It's safe to give one day more.

        :param datetime.datetime moment:
        :param datetime.timedelta oracle_interval:

        :return datetime.timedelta:
        """

        schedule_start = moment - oracle_interval * 5
        full_schedule = self.scheduler.calendar.schedule(schedule_start, moment)
        new_day = full_schedule.index[-oracle_interval.days]

        new_interval = moment.date() - new_day.date()

        return new_interval + timedelta(days=1)

    def _do_train(self, raw_data, current_moment):
        """
        Perfrorm Training

        :param raw_data:
        :type raw_data: dict
        :param current_moment:
        :type current_moment: datetime.datetime
        """
        logger.info("START training at {}".format(current_moment))
        try:
            self.oracle.train(raw_data, current_moment)
            logger.info("END training at {}".format(current_moment))
        except Exception as e:
            logger.error("SKIP training. Reason: {}".format(e))

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

        logger.info("START prediction at {}".format(current_moment))
        try:
            prediction_result = self.oracle.predict(raw_data, current_moment, target_moment)
            self.prediction_results.append(prediction_result)
            self._record_prediction(self.oracle.target_feature, prediction_result)
            self._record_actual_performance(self.oracle.target_feature, prediction_result.target_timestamp)
            logger.info("END prediction at {}".format(current_moment))
        except Exception as e:
            logger.error("SKIP prediction. Reason {}".format(e))
            return None
        else:
            return prediction_result

    def _record_actual_performance(self, feature_name, target_datetime):
        """

        :param feature_name:
        :type feature_name: str
        :param target_datetime:
        :type target_datetime: datetime.datetime
        """
        if target_datetime in self.performance:
            previous_symbols = self.performance.get_symbols(target_datetime)
            if previous_symbols is not np.nan:
                final_values = self.datasource.values_for_symbols_feature_and_time(
                    previous_symbols,
                    feature_name,
                    target_datetime
                )
                if len(final_values):
                    self.performance.add_final_values(target_datetime, final_values)
                    self.performance.save_to_hdf5(target_datetime)
            self.performance.drop_dt(target_datetime)

    def _record_prediction(self, feature_name, prediction_result):
        """
        :param feature_name:
        :type feature_name: str
        :param prediction_result:
        :type prediction_result: PredictionResult
        :return:
        """
        target_dt = prediction_result.target_timestamp
        current_datetime = prediction_result.prediction_timestamp
        prediction_symbols = np.array(prediction_result.mean_vector.index)
        initial_values = self.datasource.values_for_symbols_feature_and_time(
            prediction_symbols,
            feature_name,
            current_datetime
        )
        self.performance.add_prediction(target_dt, prediction_result.mean_vector, prediction_result.covariance_matrix)
        self.performance.add_initial_prices(target_dt, initial_values)

    def print_run_summary(self):
        logger.info("**************************")
        logger.info("**************************")
        logger.info("*** RUN OF {} FINISHED ***".format(self.name))
        logger.info("From {} to {}".format(self.simulation_start, self.simulation_end))
        logger.info(
            "Symbols: for every calendar month, we select the 400 symbols with the largest cumulative volume traded over the previous 30 days")
        logger.info("Time elapsed: {}".format(self.elapsed_time))
        logger.info("Prediction moments: ")
        logger.info(
            "{0:<50} {1:<50} {2:<50}".format("Prediction moment", "Prediction Window Start", "Prediction Window End"))
        for item in self.prediction_moments:
            logger.info("{0:<50} {1:<50} {2:<50}".format(str(item[0]), str(item[1]), str(item[2])))
        logger.info("**************************")
        logger.info("**************************")
