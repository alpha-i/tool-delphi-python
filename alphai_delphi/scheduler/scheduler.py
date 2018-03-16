import calendar
import datetime
from collections import defaultdict

import alphai_calendars as calendars
from dateutil import rrule

from alphai_delphi.configuration.schemas import SchedulingFrequencySchema
from alphai_delphi.oracle.abstract_oracle import OracleAction
from alphai_delphi.scheduler.abstract_scheduler import AbstractScheduler, SchedulingFrequencyType, ScheduleException


class Scheduler(AbstractScheduler):
    """
    The scheduler controls traning and prediction events for a controller run
    """

    def __init__(self, start_date, end_date, calendar_name, prediction_frequency, training_frequency):

        """
        :param calendar_name: Name of the exchange
        :type calendar_name: str
        """

        super().__init__(start_date, end_date, prediction_frequency, training_frequency)
        self.start_date = start_date
        self.end_date = end_date
        self.calendar_name = calendar_name
        self.calendar = calendars.get_calendar(self.calendar_name)

        frequency_schema = SchedulingFrequencySchema()
        self.prediction_frequency = frequency_schema.load(prediction_frequency).data
        self.training_frequency = frequency_schema.load(training_frequency).data

        self.schedule = defaultdict(list)
        self._init_a_schedule(self.training_frequency, OracleAction.TRAIN)
        self._init_a_schedule(self.prediction_frequency, OracleAction.PREDICT)

    def __iter__(self):
        for schedule_day in sorted(self.schedule.keys()):
            yield schedule_day, list(self.schedule[schedule_day])

    def _get_scheduled_days(self, offset):
        week_days = filter(lambda x: x.weekday() == offset,
                           rrule.rrule(rrule.DAILY, dtstart=self.start_date, until=self.end_date))
        valid_days = self.calendar.valid_days(self.start_date, self.end_date)

        result = []
        for day in week_days:
            while day not in valid_days:
                day += datetime.timedelta(days=1)
            result.append(day)
        return result

    def _init_a_schedule(self, scheduling_frequency, action):
        """
        :param scheduling_frequency:
        :type scheduling_frequency: SchedulingFrequency
        :param action:
        :type action: OracleAction
        :return:
        """
        if scheduling_frequency['frequency_type'] == SchedulingFrequencyType.MONTHLY:
            self._calculate_monthly_schedule(action, scheduling_frequency)

        if scheduling_frequency['frequency_type'] == SchedulingFrequencyType.WEEKLY:
            self._calculate_weekly_schedule(action, scheduling_frequency)

        elif scheduling_frequency['frequency_type'] == SchedulingFrequencyType.DAILY:
            self._calculate_daily_schedule(action, scheduling_frequency)

        elif scheduling_frequency['frequency_type'] == SchedulingFrequencyType.MINUTE:
            self._calculate_minute_schedule(action)

    def _calculate_minute_schedule(self, action):
        """
        Calculates the scheduling with minute resolution
        :param action:

        :return:
        """
        schedule = self.calendar.schedule(self.start_date, self.end_date)

        for day in self.calendar.valid_days(self.start_date, self.end_date):
            market_open = schedule.loc[day, "market_open"]
            market_close = schedule.loc[day, "market_close"]
            minutes = rrule.rrule(rrule.MINUTELY, dtstart=market_open, until=market_close)
            for minute in minutes:
                self.schedule[minute].append(action)

    def _calculate_daily_schedule(self, action, scheduling_frequency):
        """
        Calculates the scheduling with daily_resolution
        :param action:
        :param scheduling_frequency:
        :return:
        """
        schedule = self.calendar.schedule(self.start_date, self.end_date)
        market_days = self.calendar.valid_days(self.start_date, self.end_date)
        for day in market_days:
            market_open = schedule.loc[day, "market_open"]
            scheduled_time = market_open + datetime.timedelta(minutes=scheduling_frequency['minutes_offset'])
            self.schedule[scheduled_time].append(action)

    def _calculate_weekly_schedule(self, action, scheduling_frequency):
        """
        Calculate the schedule with a weekly resolution
        :param action:
        :param scheduling_frequency:
        :return:
        """
        schedule = self.calendar.schedule(self.start_date, self.end_date)
        scheduled_days = self._get_scheduled_days(scheduling_frequency['days_offset'])
        for day in scheduled_days:
            market_open = schedule.loc[day, "market_open"]
            scheduled_time = market_open + datetime.timedelta(minutes=scheduling_frequency['minutes_offset'])
            self.schedule[scheduled_time].append(action)

    def _calculate_monthly_schedule(self, action, scheduling_frequency):
        """
        Calculates a schedule on Montly Resolution

        :param action:
        :param scheduling_frequency:
        :return:
        """
        day_offset_of_the_month = scheduling_frequency['days_offset']
        is_offset_positive = day_offset_of_the_month >= 0

        monthly_schedule = []
        for first_day_of_the_month in rrule.rrule(rrule.MONTHLY, dtstart=self.start_date, until=self.end_date):
            if is_offset_positive:
                new_day = first_day_of_the_month + datetime.timedelta(days=day_offset_of_the_month)
            else:
                first, last = calendar.monthrange(first_day_of_the_month.year, first_day_of_the_month.month)
                new_day = first_day_of_the_month.replace(day=last)

            monthly_schedule.append(new_day)
        end_date = self.end_date if is_offset_positive else self.end_date + datetime.timedelta(days=31)
        schedule = self.calendar.schedule(self.start_date, end_date)
        valid_days = self.calendar.valid_days(self.start_date, end_date)
        scheduled_days = []
        for day in monthly_schedule:
            while day not in valid_days:
                day_offset = 1 if is_offset_positive else -1
                day += datetime.timedelta(days=day_offset)
            scheduled_days.append(day)
        for day in scheduled_days:
            market_open = schedule.loc[day, "market_open"]
            scheduled_time = market_open + datetime.timedelta(minutes=scheduling_frequency['minutes_offset'])
            self.schedule[scheduled_time].append(action)

    def get_first_valid_target(self, moment, interval):
        schedule = self.calendar.schedule(self.start_date, self.end_date)
        target = moment + interval

        while not self.calendar.open_at_time(schedule, target, include_close=True):
            target += datetime.timedelta(days=1)
            if target > self.end_date:
                raise ScheduleException("Target outside of scheduling window")

        return target

    def get_event(self, minute):
        events = self.schedule.get(minute)
        return list(events) if events else []
