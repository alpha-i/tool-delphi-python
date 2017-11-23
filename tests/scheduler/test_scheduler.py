from delphi.oracle import OracleAction
from delphi.scheduler import Scheduler
from delphi.scheduler.abstract_scheduler import SchedulingFrequency, SchedulingFrequencyType
from pandas import Timestamp

import datetime
import pytz


def test_weekly_prediction_and_training_scheduler():
    test_scheduler = Scheduler(
        start_date=datetime.datetime(2017, 11, 6, tzinfo=pytz.utc),
        end_date=datetime.datetime(2017, 11, 21, tzinfo=pytz.utc),
        exchange_name='NYSE',
        prediction_horizon=datetime.timedelta(days=1),
        prediction_frequency=SchedulingFrequency(
            frequency_type=SchedulingFrequencyType.WEEKLY, days_offset=0, minutes_offset=30
        ),  # every Monday, 30m after market start
        training_frequency=SchedulingFrequency(
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
        exchange_name='NYSE',
        prediction_horizon=datetime.timedelta(days=1),
        prediction_frequency=SchedulingFrequency(
            frequency_type=SchedulingFrequencyType.WEEKLY, days_offset=1, minutes_offset=30
        ),  # every Tuesday, 30m after market start
        training_frequency=SchedulingFrequency(
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
        exchange_name='NYSE',
        prediction_horizon=datetime.timedelta(days=1),
        prediction_frequency=SchedulingFrequency(
            frequency_type=SchedulingFrequencyType.WEEKLY, days_offset=0, minutes_offset=15
        ),  # every Monday, 15m after market start
        training_frequency=SchedulingFrequency(
            frequency_type=SchedulingFrequencyType.WEEKLY, days_offset=0, minutes_offset=15
        ),  # every Monday, 15m after market start
    )

    events = test_scheduler.get_event(datetime.datetime(2017, 12, 25, 14, 45, tzinfo=pytz.UTC))
    assert not events

    events = test_scheduler.get_event(datetime.datetime(2018, 1, 1, 14, 45, tzinfo=pytz.UTC))
    assert not events

    events = test_scheduler.get_event(datetime.datetime(2017, 12, 26, 14, 45, tzinfo=pytz.UTC))
    assert set(events) == {OracleAction.PREDICT, OracleAction.TRAIN}

    events = test_scheduler.get_event(datetime.datetime(2018, 1, 2, 14, 45, tzinfo=pytz.UTC))
    assert set(events) == {OracleAction.PREDICT, OracleAction.TRAIN}


def test_daily_prediction_scheduler():
    test_scheduler = Scheduler(
        start_date=datetime.datetime(2017, 12, 20, tzinfo=pytz.utc),
        end_date=datetime.datetime(2018, 1, 3, tzinfo=pytz.utc),
        exchange_name='NYSE',
        prediction_horizon=datetime.timedelta(days=1),
        prediction_frequency=SchedulingFrequency(
            frequency_type=SchedulingFrequencyType.DAILY, minutes_offset=15
        ),  # every day, 15m after market start
        training_frequency=SchedulingFrequency(
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
        exchange_name='NYSE',
        prediction_horizon=datetime.timedelta(days=1),
        prediction_frequency=SchedulingFrequency(frequency_type=SchedulingFrequencyType.MINUTE),
        training_frequency=SchedulingFrequency(frequency_type=SchedulingFrequencyType.MINUTE)
    )

    assert sorted(list(test_scheduler.schedule.keys()))[0] == datetime.datetime(2017, 12, 20, 14, 30, tzinfo=pytz.UTC)
    assert sorted(list(test_scheduler.schedule.keys()))[-1] == datetime.datetime(2017, 12, 20, 21, 0, tzinfo=pytz.UTC)


def test_scheduler_works_as_iterator():
    test_scheduler = Scheduler(
        start_date=datetime.datetime(2017, 11, 6, tzinfo=pytz.utc),
        end_date=datetime.datetime(2017, 11, 21, tzinfo=pytz.utc),
        exchange_name='NYSE',
        prediction_horizon=datetime.timedelta(days=1),
        prediction_frequency=SchedulingFrequency(
            frequency_type=SchedulingFrequencyType.WEEKLY, days_offset=1, minutes_offset=30
        ),  # every Tuesday, 30m after market start
        training_frequency=SchedulingFrequency(
            frequency_type=SchedulingFrequencyType.WEEKLY, days_offset=1, minutes_offset=30
        ),  # every Tuesday, 30m after market start
    )

    for day, events in test_scheduler:
        assert isinstance(day, datetime.datetime)
        assert isinstance(events, list)


def test_scheduler_checks_for_valid_prediction_target():
    test_scheduler = Scheduler(
        start_date=datetime.datetime(2017, 12, 20, tzinfo=pytz.utc),
        end_date=datetime.datetime(2018, 1, 3, tzinfo=pytz.utc),
        exchange_name='NYSE',
        prediction_horizon=datetime.timedelta(days=1),
        prediction_frequency=SchedulingFrequency(
            frequency_type=SchedulingFrequencyType.DAILY, minutes_offset=15
        ),  # every day, 15m after market start
        training_frequency=SchedulingFrequency(
            frequency_type=SchedulingFrequencyType.DAILY,
            minutes_offset=15
        ),  # every day, 30m after market start
    )
    prediction_target = test_scheduler.get_first_valid_target(
        moment=datetime.datetime(2017, 12, 24, 16, 0, tzinfo=pytz.UTC),
        interval=datetime.timedelta(days=1)
    )

    assert prediction_target == datetime.datetime(2017, 12, 26, 16, 0, tzinfo=pytz.UTC)


def test_scheduler_checks_for_valid_prediction_target_with_early_close():
    test_scheduler = Scheduler(
        start_date=datetime.datetime(2017, 11, 1, tzinfo=pytz.utc),
        end_date=datetime.datetime(2018, 1, 3, tzinfo=pytz.utc),
        exchange_name='NYSE',
        prediction_horizon=datetime.timedelta(days=1),
        prediction_frequency=SchedulingFrequency(
            frequency_type=SchedulingFrequencyType.DAILY, minutes_offset=15
        ),  # every day, 15m after market start
        training_frequency=SchedulingFrequency(
            frequency_type=SchedulingFrequencyType.DAILY,
            minutes_offset=15
        ),  # every day, 30m after market start
    )
    prediction_target = test_scheduler.get_first_valid_target(
        moment=datetime.datetime(2017, 11, 23, 16, 0, tzinfo=pytz.UTC),  # Thanksgiving is an early close (1PM) day
        interval=datetime.timedelta(days=1)
    )

    assert prediction_target == datetime.datetime(2017, 11, 24, 16, 0, tzinfo=pytz.UTC)
