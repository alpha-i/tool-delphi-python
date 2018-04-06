import pandas as pd
from pandas import Timestamp

from alphai_delphi.scheduler.scheduler import Scheduler
from alphai_delphi.scheduler.abstract_scheduler import SchedulingFrequencyType
from alphai_delphi.oracle.abstract_oracle import OracleAction

import datetime
import pytz


def test_weekly_prediction_and_training_scheduler():
    test_scheduler = Scheduler(
        start_date=datetime.datetime(2017, 11, 6, tzinfo=pytz.utc),
        end_date=datetime.datetime(2017, 11, 21, tzinfo=pytz.utc),
        calendar_name='NYSE',
        prediction_frequency=dict(
            frequency_type=SchedulingFrequencyType.WEEKLY, days_offset=0, minutes_offset=30
        ),  # every Monday, 30m after market start
        training_frequency=dict(
            frequency_type=SchedulingFrequencyType.WEEKLY, days_offset=0, minutes_offset=30
        ),  # every Monday, 30m after market start
    )
    events = test_scheduler.get_event(datetime.datetime(2017, 11, 6, 15, 0, tzinfo=pytz.UTC))
    assert set(events) == {OracleAction.PREDICT, OracleAction.TRAIN}

    events = test_scheduler.get_event(datetime.datetime(2017, 11, 13, 15, 0, tzinfo=pytz.UTC))
    assert set(events) == {OracleAction.PREDICT, OracleAction.TRAIN}


def test_weekly_prediction_scheduler_starting_on_tuesday():
    test_scheduler = Scheduler(
        start_date=datetime.datetime(2017, 11, 6, tzinfo=pytz.utc),
        end_date=datetime.datetime(2017, 11, 21, tzinfo=pytz.utc),
        calendar_name='NYSE',
        prediction_frequency=dict(
            frequency_type=SchedulingFrequencyType.WEEKLY, days_offset=1, minutes_offset=30
        ),  # every Tuesday, 30m after market start
        training_frequency=dict(
            frequency_type=SchedulingFrequencyType.WEEKLY, days_offset=1, minutes_offset=30
        ),  # every Tuesday, 30m after market start
    )

    events = test_scheduler.get_event(datetime.datetime(2017, 11, 6, 15, 0, tzinfo=pytz.UTC))
    assert not events

    events = test_scheduler.get_event(datetime.datetime(2017, 11, 7, 15, 0, tzinfo=pytz.UTC))
    assert set(events) == {OracleAction.TRAIN, OracleAction.PREDICT}

    events = test_scheduler.get_event(datetime.datetime(2017, 11, 14, 15, 0, tzinfo=pytz.UTC))
    assert set(events) == {OracleAction.PREDICT, OracleAction.TRAIN}


def test_weekly_prediction_scheduler_with_a_public_holiday():
    test_scheduler = Scheduler(
        start_date=datetime.datetime(2017, 12, 20, tzinfo=pytz.utc),
        end_date=datetime.datetime(2018, 1, 3, tzinfo=pytz.utc),
        calendar_name='NYSE',
        prediction_frequency=dict(
            frequency_type=SchedulingFrequencyType.WEEKLY, days_offset=0, minutes_offset=15
        ),  # every Monday, 15m after market start
        training_frequency=dict(
            frequency_type=SchedulingFrequencyType.WEEKLY, days_offset=0, minutes_offset=15
        ),  # every Monday, 15m after market start
    )

    events = test_scheduler.get_event(datetime.datetime(2017, 12, 25, 14, 45, tzinfo=pytz.UTC))
    assert not events

    events = test_scheduler.get_event(datetime.datetime(2018, 1, 1, 14, 45, tzinfo=pytz.UTC))
    assert not events

    events = test_scheduler.get_event(datetime.datetime(2017, 12, 26, 14, 45, tzinfo=pytz.UTC))
    assert events == [OracleAction.TRAIN, OracleAction.PREDICT]

    events = test_scheduler.get_event(datetime.datetime(2018, 1, 2, 14, 45, tzinfo=pytz.UTC))
    assert events == [OracleAction.TRAIN, OracleAction.PREDICT]


def test_daily_prediction_scheduler():
    test_scheduler = Scheduler(
        start_date=datetime.datetime(2017, 12, 20, tzinfo=pytz.utc),
        end_date=datetime.datetime(2018, 1, 3, tzinfo=pytz.utc),
        calendar_name='NYSE',
        prediction_frequency=dict(
            frequency_type=SchedulingFrequencyType.DAILY, minutes_offset=15
        ),  # every day, 15m after market start
        training_frequency=dict(
            frequency_type=SchedulingFrequencyType.DAILY,
            minutes_offset=15
        ),  # every day, 30m after market start
    )
    assert sorted(list(test_scheduler.schedule.keys())) == [
        Timestamp('2017-12-20 14:45:00+0000', tz='UTC'),
        Timestamp('2017-12-21 14:45:00+0000', tz='UTC'),
        Timestamp('2017-12-22 14:45:00+0000', tz='UTC'),
        Timestamp('2017-12-26 14:45:00+0000', tz='UTC'),
        Timestamp('2017-12-27 14:45:00+0000', tz='UTC'),
        Timestamp('2017-12-28 14:45:00+0000', tz='UTC'),
        Timestamp('2017-12-29 14:45:00+0000', tz='UTC'),
        Timestamp('2018-01-02 14:45:00+0000', tz='UTC'),
        Timestamp('2018-01-03 14:45:00+0000', tz='UTC'),
    ]


def test_minute_prediction_scheduler():
    test_scheduler = Scheduler(
        start_date=datetime.datetime(2017, 12, 20, 15, 0, tzinfo=pytz.utc),
        end_date=datetime.datetime(2017, 12, 20, 18, 0, tzinfo=pytz.utc),
        calendar_name='NYSE',
        prediction_frequency=dict(frequency_type=SchedulingFrequencyType.MINUTE),
        training_frequency=dict(frequency_type=SchedulingFrequencyType.MINUTE)
    )

    assert sorted(list(test_scheduler.schedule.keys()))[0] == datetime.datetime(2017, 12, 20, 14, 30, tzinfo=pytz.UTC)
    assert sorted(list(test_scheduler.schedule.keys()))[-1] == datetime.datetime(2017, 12, 20, 21, 0, tzinfo=pytz.UTC)


def test_scheduler_works_as_iterator():
    test_scheduler = Scheduler(
        start_date=datetime.datetime(2017, 11, 6, tzinfo=pytz.utc),
        end_date=datetime.datetime(2017, 11, 21, tzinfo=pytz.utc),
        calendar_name='NYSE',
        prediction_frequency=dict(
            frequency_type=SchedulingFrequencyType.WEEKLY, days_offset=1, minutes_offset=30
        ),  # every Tuesday, 30m after market start
        training_frequency=dict(
            frequency_type=SchedulingFrequencyType.WEEKLY, days_offset=1, minutes_offset=30
        ),  # every Tuesday, 30m after market start
    )

    for day, events in test_scheduler:
        assert isinstance(day, datetime.datetime)
        assert isinstance(events, list)


def test_scheduler_monthly():
    test_scheduler = Scheduler(
        start_date=datetime.datetime(2017, 11, 1, tzinfo=pytz.utc),
        end_date=datetime.datetime(2018, 2, 3, tzinfo=pytz.utc),
        calendar_name='NYSE',
        prediction_frequency=dict(frequency_type=SchedulingFrequencyType.MONTHLY, days_offset=0),
        training_frequency=dict(frequency_type=SchedulingFrequencyType.MONTHLY, days_offset=0),
    )

    assert sorted(list(test_scheduler.schedule.keys())) == [
        Timestamp('2017-11-01 13:30:00+0000', tz='UTC'),
        Timestamp('2017-12-01 14:30:00+0000', tz='UTC'),
        Timestamp('2018-01-02 14:30:00+0000', tz='UTC'),
        Timestamp('2018-02-01 14:30:00+0000', tz='UTC'),
    ]

    test_scheduler = Scheduler(
        start_date=datetime.datetime(2017, 11, 1, tzinfo=pytz.utc),
        end_date=datetime.datetime(2018, 2, 3, tzinfo=pytz.utc),
        calendar_name='NYSE',
        prediction_frequency=dict(frequency_type=SchedulingFrequencyType.MONTHLY, days_offset=-1),
        training_frequency=dict(frequency_type=SchedulingFrequencyType.MONTHLY, days_offset=-1),
    )

    assert sorted(list(test_scheduler.schedule.keys())) == [
        Timestamp('2017-11-30 14:30:00+0000', tz='UTC'),
        Timestamp('2017-12-29 14:30:00+0000', tz='UTC'),
        Timestamp('2018-01-31 14:30:00+0000', tz='UTC'),
        Timestamp('2018-02-28 14:30:00+0000', tz='UTC'),
    ]


def test_scheduler_end_of_month():

    test_scheduler = Scheduler(
        start_date=datetime.datetime(2006, 11, 15, tzinfo=pytz.utc),
        end_date=datetime.datetime(2017, 9, 12, tzinfo=pytz.utc),
        calendar_name='JSE',
        prediction_frequency=dict(frequency_type=SchedulingFrequencyType.MONTHLY, days_offset=-1),
        training_frequency=dict(frequency_type=SchedulingFrequencyType.MONTHLY, days_offset=-1),
    )

    end_of_month_days = ['2017-09-29', '2017-08-31', '2017-07-31', '2017-06-30', '2017-05-31', '2017-04-28',
                         '2017-03-31', '2017-02-28', '2017-01-31', '2016-12-30', '2016-11-30', '2016-10-31',
                         '2016-09-30', '2016-08-31', '2016-07-29', '2016-06-30', '2016-05-31', '2016-04-29',
                         '2016-03-31', '2016-02-29', '2016-01-29', '2015-12-31', '2015-11-30', '2015-10-30',
                         '2015-09-30', '2015-08-31', '2015-07-31', '2015-06-30', '2015-05-29', '2015-04-30',
                         '2015-03-31', '2015-02-27', '2015-01-30', '2014-12-31', '2014-11-28', '2014-10-31',
                         '2014-09-30', '2014-08-29', '2014-07-31', '2014-06-30', '2014-05-30', '2014-04-30',
                         '2014-03-31', '2014-02-28', '2014-01-31', '2013-12-31', '2013-11-29', '2013-10-31',
                         '2013-09-30', '2013-08-30', '2013-07-31', '2013-06-28', '2013-05-31', '2013-04-30',
                         '2013-03-28', '2013-02-28', '2013-01-31', '2012-12-31', '2012-11-30', '2012-10-31',
                         '2012-09-28', '2012-08-31', '2012-07-31', '2012-06-29', '2012-05-31', '2012-04-30',
                         '2012-03-30', '2012-02-29', '2012-01-31', '2011-12-30', '2011-11-30', '2011-10-31',
                         '2011-09-30', '2011-08-31', '2011-07-29', '2011-06-30', '2011-05-31', '2011-04-29',
                         '2011-03-31', '2011-02-28', '2011-01-31', '2010-12-31', '2010-11-30', '2010-10-29',
                         '2010-09-30', '2010-08-31', '2010-07-30', '2010-06-30', '2010-05-31', '2010-04-30',
                         '2010-03-31', '2010-02-26', '2010-01-29', '2009-12-31', '2009-11-30', '2009-10-30',
                         '2009-09-30', '2009-08-31', '2009-07-31', '2009-06-30', '2009-05-29', '2009-04-30',
                         '2009-03-31', '2009-02-27', '2009-01-30', '2008-12-31', '2008-11-28', '2008-10-31',
                         '2008-09-30', '2008-08-29', '2008-07-31', '2008-06-30', '2008-05-30', '2008-04-30',
                         '2008-03-31', '2008-02-29', '2008-01-31', '2007-12-31', '2007-11-30', '2007-10-31',
                         '2007-09-28', '2007-08-31', '2007-07-31', '2007-06-29', '2007-05-31', '2007-04-30',
                         '2007-03-30', '2007-02-28', '2007-01-31', '2006-12-29', '2006-11-30']

    def as_timestamp(x):
        return pd.Timestamp(x, tz='UTC').replace(hour=7)

    expected_timestamp_list = sorted(map(as_timestamp, end_of_month_days))
    returned_timestamp_list = sorted(list(test_scheduler.schedule.keys()))
    assert len(returned_timestamp_list) == len(expected_timestamp_list)
    assert returned_timestamp_list == expected_timestamp_list


def test_february():

    test_scheduler = Scheduler(
        start_date=datetime.datetime(2007, 1, 30, tzinfo=pytz.utc),
        end_date=datetime.datetime(2007, 4, 1, tzinfo=pytz.utc),
        calendar_name='JSE',
        prediction_frequency=dict(frequency_type=SchedulingFrequencyType.MONTHLY, days_offset=-1),
        training_frequency=dict(frequency_type=SchedulingFrequencyType.MONTHLY, days_offset=-1),
    )

    end_of_month_days = ['2007-04-30', '2007-03-30', '2007-02-28', '2007-01-31']

    def as_timestamp(x):
        return pd.Timestamp(x, tz='UTC').replace(hour=7)

    expected_timestamp_list = sorted(map(as_timestamp, end_of_month_days))
    returned_timestamp_list = sorted(list(test_scheduler.schedule.keys()))
    assert len(returned_timestamp_list) == len(expected_timestamp_list)
    assert returned_timestamp_list == expected_timestamp_list
