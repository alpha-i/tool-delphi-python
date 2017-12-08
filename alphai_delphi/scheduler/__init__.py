import datetime
from collections import defaultdict

import pandas_market_calendars as calendar
from dateutil import rrule

from alphai_delphi.oracle import OracleAction
from alphai_delphi.scheduler.abstract_scheduler import AbstractScheduler, SchedulingFrequencyType, ScheduleException


class Scheduler(AbstractScheduler):
    """
    The scheduler controls traning and prediction events for a controller run
    """

    def __init__(self, start_date, end_date, exchange_name,
                 prediction_frequency, training_frequency,
                 prediction_horizon):

        """
        :param exchange_name: Name of the exchange
        :type exchange_name: str
        """

        super().__init__(start_date, end_date, prediction_frequency, training_frequency, prediction_horizon)
        self.start_date = start_date
        self.end_date = end_date
        self.exchange_name = exchange_name

        self.prediction_frequency = prediction_frequency
        self.training_frequency = training_frequency

        self.prediction_horizon = prediction_horizon

        self.schedule = defaultdict(list)
        self._init_a_schedule(self.training_frequency, OracleAction.TRAIN)
        self._init_a_schedule(self.prediction_frequency, OracleAction.PREDICT)

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
            scheduled_days = self._get_scheduled_days(
                exchange_calendar, schedule.days_offset, self.start_date,
                self.end_date)

            for day in scheduled_days:
                market_open = exchange_schedule.loc[day, "market_open"]
                scheduled_time = market_open + datetime.timedelta(minutes=schedule.minutes_offset)
                self.schedule[scheduled_time].append(action)

        elif schedule.frequency_type == SchedulingFrequencyType.DAILY:
            market_days = exchange_calendar.valid_days(self.start_date, self.end_date)
            for day in market_days:
                market_open = exchange_schedule.loc[day, "market_open"]
                scheduled_time = market_open + datetime.timedelta(minutes=schedule.minutes_offset)
                self.schedule[scheduled_time].append(action)

        elif schedule.frequency_type == SchedulingFrequencyType.MINUTE:
            for day in exchange_calendar.valid_days(self.start_date, self.end_date):
                market_open = exchange_schedule.loc[day, "market_open"]
                market_close = exchange_schedule.loc[day, "market_close"]
                minutes = rrule.rrule(rrule.MINUTELY, dtstart=market_open, until=market_close)
                for minute in minutes:
                    self.schedule[minute].append(action)

    def get_first_valid_target(self, moment, interval):
        exchange_calendar = calendar.get_calendar(self.exchange_name)
        exchange_schedule = exchange_calendar.schedule(self.start_date, self.end_date)
        target = moment + interval

        while not exchange_calendar.open_at_time(exchange_schedule, target):
            target += datetime.timedelta(days=1)
            if target > self.end_date:
                raise ScheduleException("Target outside of scheduling window")

        return target

    def get_event(self, minute):
        events = self.schedule.get(minute)
        return list(events) if events else []
