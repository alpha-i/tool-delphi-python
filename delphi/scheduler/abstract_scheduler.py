from enum import Enum


class SchedulingFrequency(Enum):
    DAILY = 0
    MINUTE = 1
    WEEKLY = 2
    NEVER = 3


class Scheduler:
    """
    The scheduler controls traning and prediction events for a controller run
    """
    def __init__(self, start_date, end_date,
                 training_frequency, prediction_offset,
                 prediction_frequency, prediction_horizon):
        """
        :param start_date:
        :type start_date: datetime.datetime
        :param end_date:
        :type end_date: datetime.datetime
        :param training_frequency:
        :type training_frequency: SchedulingFrequency
        :param prediction_frequency:
        :type prediction_frequency: SchedulingFrequency
        :param prediction_horizon:
        :type prediction_horizon: datetime.timedelta
        :param prediction_offset:
        :type prediction_offset: datetime.timedelta
        """
        self.start_date = start_date
        self.end_date = end_date
        self.training_frequency = training_frequency
        self.prediction_frequency = prediction_frequency
        self.prediction_horizon = prediction_horizon
        self.prediction_offset = prediction_offset

        # TODO: make the actual schedule taking market times into account
        self.schedule = {}

    def get_event(self, minute):
        """
        Given a minute, give back the list of action(s) that the oracle is supposed to perform

        :param minute: Minute in time to get an event for
        :type minute: datetime.datetime
        :return: List of actions to be performed by the oracle
        :rtype: List[OracleActions]
        """

        return self.schedule.get(minute)
