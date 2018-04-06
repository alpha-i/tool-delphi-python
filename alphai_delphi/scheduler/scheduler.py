import calendar
import datetime
from collections import defaultdict

import alphai_calendars as calendars
from dateutil import rrule

from alphai_delphi.configuration.schemas import SchedulingFrequencySchema, InvalidConfiguration
from alphai_delphi.oracle.abstract_oracle import OracleAction
from alphai_delphi.scheduler.abstract_scheduler import AbstractScheduler, SchedulingFrequencyType, ScheduleException


class Scheduler(AbstractScheduler):
    """
    The scheduler controls traning and prediction events for a controller run
    """

    def __init__(self, start_date, end_date, calendar_name, prediction_frequency, training_frequency):
        """
        :param datetime.datetime start_date:
        :param datetime.datetime end_date:
        :param str calendar_name:
        :param dict prediction_frequency:
        :param dict training_frequency:
        """

        super().__init__(start_date, end_date, prediction_frequency, training_frequency)
        self.start_date = start_date
        self.end_date = end_date
        self.calendar_name = calendar_name
        self.calendar = calendars.get_calendar(self.calendar_name)

        self.prediction_frequency = self._validate_frequency(prediction_frequency)
        self.training_frequency = self._validate_frequency(training_frequency)

        self.schedule = defaultdict(list)
        self._init_a_schedule(self.training_frequency, OracleAction.TRAIN)
        self._init_a_schedule(self.prediction_frequency, OracleAction.PREDICT)

    def _validate_frequency(self, frequency):
        """
        Validates the frequency against the schema
        :param frequency:
        :return:
        """
        frequency_schema = SchedulingFrequencySchema()

        data, errors = frequency_schema.dump(frequency)
        if errors:
            raise InvalidConfiguration("Schema loading Failure on scheduler {}".format(errors))

        data, errors = frequency_schema.load(data)

        if errors:
            raise InvalidConfiguration("Schema creation Failure on scheduler {}".format(errors))

        return data

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
        computed_moments = []
        if scheduling_frequency['frequency_type'] == SchedulingFrequencyType.MONTHLY:
            computed_moments = self._calculate_monthly_schedule(scheduling_frequency)

        if scheduling_frequency['frequency_type'] == SchedulingFrequencyType.WEEKLY:
            computed_moments = self._calculate_weekly_schedule(scheduling_frequency)

        elif scheduling_frequency['frequency_type'] == SchedulingFrequencyType.DAILY:
            computed_moments = self._calculate_daily_schedule(scheduling_frequency)

        elif scheduling_frequency['frequency_type'] == SchedulingFrequencyType.MINUTE:
            computed_moments = self._calculate_minute_schedule()

        for moment in computed_moments:
            self.schedule[moment].append(action)

    def _calculate_minute_schedule(self):
        """
        Calculates the scheduling with minute resolution

        :return:
        """
        schedule = self.calendar.schedule(self.start_date, self.end_date)

        computed_moments = []
        for day in self.calendar.valid_days(self.start_date, self.end_date):
            market_open = schedule.loc[day, "market_open"]
            market_close = schedule.loc[day, "market_close"]
            minutes = rrule.rrule(rrule.MINUTELY, dtstart=market_open, until=market_close)
            computed_moments = computed_moments + list(minutes)

        return computed_moments

    def _calculate_daily_schedule(self, scheduling_frequency):
        """
        Calculates the scheduling with daily_resolution
        :param scheduling_frequency:
        :return:
        """
        schedule = self.calendar.schedule(self.start_date, self.end_date)
        market_days = self.calendar.valid_days(self.start_date, self.end_date)
        computed_moments = []
        for day in market_days:
            market_open = schedule.loc[day, "market_open"]
            scheduled_time = market_open + datetime.timedelta(minutes=scheduling_frequency['minutes_offset'])
            computed_moments.append(scheduled_time)

        return computed_moments

    def _calculate_weekly_schedule(self, scheduling_frequency):
        """
        Calculate the schedule with a weekly resolution
        :param scheduling_frequency:
        :return : list of pd.Timestamp
        """
        schedule = self.calendar.schedule(self.start_date, self.end_date)
        scheduled_days = self._get_scheduled_days(scheduling_frequency['days_offset'])
        computed_moments = []
        for day in scheduled_days:
            market_open = schedule.loc[day, "market_open"]
            scheduled_time = market_open + datetime.timedelta(minutes=scheduling_frequency['minutes_offset'])
            computed_moments.append(scheduled_time)

        return computed_moments

    def _calculate_monthly_schedule(self, scheduling_frequency):
        """
        Calculates a schedule on Monthly Resolution

        :param scheduling_frequency:
        :return: list of pd.Timestamp
        """
        day_offset_of_the_month = scheduling_frequency['days_offset']
        is_offset_positive = day_offset_of_the_month >= 0

        monthly_schedule = []
        start_date = self.start_date.replace(day=1)
        for first_day_of_the_month in rrule.rrule(rrule.MONTHLY, dtstart=start_date, until=self.end_date):
            if is_offset_positive:
                new_day = first_day_of_the_month + datetime.timedelta(days=day_offset_of_the_month)
            else:
                first, last = calendar.monthrange(first_day_of_the_month.year, first_day_of_the_month.month)
                new_day = first_day_of_the_month.replace(day=last)

            monthly_schedule.append(new_day)

        if is_offset_positive:
            end_date = self.end_date
        else:
            first, last = calendar.monthrange(self.end_date.year, self.end_date.month)
            end_date = self.end_date.replace(day=last)

        schedule = self.calendar.schedule(self.start_date, end_date)
        valid_days = self.calendar.valid_days(self.start_date, end_date)
        day_offset = 1 if is_offset_positive else -1

        computed_moments = []
        for day in monthly_schedule:
            while day not in valid_days:
                day += datetime.timedelta(days=day_offset)

            market_open = schedule.loc[day, "market_open"]
            scheduled_time = market_open + datetime.timedelta(minutes=scheduling_frequency['minutes_offset'])
            computed_moments.append(scheduled_time)

        return computed_moments

    def get_event(self, minute):
        events = self.schedule.get(minute)
        return list(events) if events else []
