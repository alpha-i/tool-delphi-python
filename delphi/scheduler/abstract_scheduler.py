import datetime
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from enum import Enum

import pandas_market_calendars as calendar
from dateutil import rrule

from delphi.oracle import OracleAction


class SchedulingFrequencyType(Enum):
    DAILY = 0
    MINUTE = 1
    WEEKLY = 2
    NEVER = 3


class SchedulingFrequency:
    def __init__(self, frequency_type, days_offset=0, minutes_offset=0):
        """
        :param frequency_type: Type of the scheduling frequency
        :type frequency_type: SchedulingFrequencyType
        :param days_offset:
        :type days_offset: int
        :param minutes_offset:
        :type minutes_offset: int
        """
        self.frequency_type = frequency_type
        self.minutes_offset = minutes_offset
        self.days_offset = days_offset


class AbstractScheduler(metaclass=ABCMeta):
    def __init__(self, start_date, end_date,
                 prediction_scheduling, training_scheduling,
                 prediction_horizon):
        """
        :param start_date:
        :type start_date: datetime.datetime
        :param end_date:
        :type end_date: datetime.datetime
        :param training_scheduling:
        :type training_scheduling: SchedulingFrequency
        :param prediction_scheduling:
        :type prediction_scheduling: SchedulingFrequency
        :param prediction_horizon:
        :type prediction_horizon: datetime.timedelta
        """
        self.start_date = start_date
        self.end_date = end_date
        self.prediction_scheduling = prediction_scheduling
        self.training_scheduling = training_scheduling
        self.prediction_horizon = prediction_horizon

    @abstractmethod
    def __iter__(self):
        raise NotImplemented

    @abstractmethod
    def get_event(self, minute):
        raise NotImplemented


class Scheduler(AbstractScheduler):
    """
    The scheduler controls traning and prediction events for a controller run
    """

    def __init__(self, start_date, end_date, exchange_name,
                 prediction_scheduling, training_scheduling,
                 prediction_horizon):

        """
        :param exchange_name: Name of the exchange
        :type exchange_name: str
        """

        super().__init__(start_date, end_date, prediction_scheduling, training_scheduling, prediction_horizon)
        self.start_date = start_date
        self.end_date = end_date
        self.exchange_name = exchange_name

        self.prediction_scheduling = prediction_scheduling
        self.training_scheduling = training_scheduling

        self.prediction_horizon = prediction_horizon

        self.schedule = defaultdict(set)
        self._init_a_schedule(self.training_scheduling, OracleAction.TRAIN)
        self._init_a_schedule(self.prediction_scheduling, OracleAction.PREDICT)

    def __iter__(self):
        for schedule_day in sorted(self.schedule.keys()):
            yield schedule_day, list(self.schedule[schedule_day])

    def _get_scheduled_days(self, calendar, offset, start_date, end_date):
        week_days = filter(lambda x: x.weekday() == offset,
                           rrule.rrule(rrule.DAILY, dtstart=start_date, until=end_date))
        valid_days = calendar.valid_days(start_date, end_date)

        result = []
        for day in week_days:
            while day not in valid_days:
                day += datetime.timedelta(days=1)
            result.append(day)
        return result

    def _init_a_schedule(self, schedule, action):
        """
        :param schedule:
        :type schedule: SchedulingFrequency
        :param action:
        :type action: OracleAction
        :return:
        """
        exchange_calendar = calendar.get_calendar(self.exchange_name)
        exchange_schedule = exchange_calendar.schedule(self.start_date, self.end_date)
        if schedule.frequency_type == SchedulingFrequencyType.WEEKLY:
            # get a list of all the days where the action should take place
            # (.weekday is the same as the offset: 0 is a Monday)
            training_days = self._get_scheduled_days(exchange_calendar, schedule.days_offset, self.start_date,
                                                     self.end_date)

            for day in training_days:
                market_open = exchange_schedule.loc[day, "market_open"]
                scheduled_time = market_open + datetime.timedelta(minutes=schedule.minutes_offset)
                self.schedule[scheduled_time].add(action)

        elif schedule.frequency_type == SchedulingFrequencyType.DAILY:
            market_days = exchange_calendar.valid_days(self.start_date, self.end_date)
            for day in market_days:
                market_open = exchange_schedule.loc[day, "market_open"]
                scheduled_time = market_open + datetime.timedelta(minutes=schedule.minutes_offset)
                self.schedule[scheduled_time].add(action)

        elif schedule.frequency_type == SchedulingFrequencyType.MINUTE:
            for day in exchange_calendar.valid_days(self.start_date, self.end_date):
                market_open = exchange_schedule.loc[day, "market_open"]
                market_close = exchange_schedule.loc[day, "market_close"]
                minutes = rrule.rrule(rrule.MINUTELY, dtstart=market_open, until=market_close)
                for minute in minutes:
                    self.schedule[minute].add(action)

    def get_event(self, minute):
        """
        Given a minute, give back the list of action(s) that the oracle is supposed to perform

        :param minute: Minute in time to get an event for
        :type minute: datetime.datetime
        :return: List of actions to be performed by the oracle
        :rtype: List[OracleActions]
        """
        events = self.schedule.get(minute)
        return list(events) if events else []
