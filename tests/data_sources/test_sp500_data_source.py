import datetime

import psycopg2
import pytest
import pytz
import pandas as pd

from alphai_delphi.data_source.sp500_data_source import SP500DataSource

data_source_config = {
    "filename": "",
    "exchange": "NYSE",
    "start": datetime.datetime(1999, 1, 1, tzinfo=pytz.utc),
    "end": datetime.datetime(1999, 3, 1, tzinfo=pytz.utc)
}


"""
Test data:

Minutes:
    AAPL	100	50	50	50	200	2017-01-15 14:30:00+00
    MSFT	100	50	50	50	200	2017-01-15 14:30:00+00
    AAPL	100	50	50	50	200	2017-01-14 14:30:00+00
    MSFT	100	50	50	50	200	2017-01-14 14:30:00+00
    AAPL	200	50	50	50	200	2017-01-13 14:30:00+00
    MSFT	200	50	50	50	200	2017-01-13 14:30:00+00
    AAPL	200	50	50	50	200	2017-01-12 14:30:00+00
    MSFT	200	50	50	50	200	2017-01-12 14:30:00+00
    AAPL	400	50	50	50	200	2017-01-11 14:30:00+00
    MSFT	400	50	50	50	200	2017-01-11 14:30:00+00
    AAPL	400	50	50	50	200	2017-01-10 14:30:00+00
    MSFT	400	50	50	50	200	2017-01-10 14:30:00+00
    
Adjustments:
    AAPL	1	1	    2050-01-01
    AAPL	1	0.5	    2017-01-13
    MSFT	1	0.5	    2017-01-12
    AAPL	1	0.25	2017-01-11
    
Resulting adjusted data for 'open' should be 100 all the way     
"""


class MockSP500DataSource(SP500DataSource):
    """
    A data source class with mocked methods to avoid sql calls when testing
    """
    def __init__(self, configuration):
        pass

    def _query_minutes(self, moment, interval):
        return [{'timestamp': datetime.datetime(2017, 1, 15, 14, 30,
                                                tzinfo=psycopg2.tz.FixedOffsetTimezone(offset=0, name=None)),
                 'symbol': 'AAPL', 'high': 50.0, 'close': 50.0, 'id': 3526, 'volume': 200, 'low': 50.0, 'open': 100.0},
                {'timestamp': datetime.datetime(2017, 1, 15, 14, 30,
                                                tzinfo=psycopg2.tz.FixedOffsetTimezone(offset=0, name=None)),
                 'symbol': 'MSFT', 'high': 50.0, 'close': 50.0, 'id': 3532, 'volume': 200, 'low': 50.0, 'open': 100.0},
                {'timestamp': datetime.datetime(2017, 1, 14, 14, 30,
                                                tzinfo=psycopg2.tz.FixedOffsetTimezone(offset=0, name=None)),
                 'symbol': 'AAPL', 'high': 50.0, 'close': 50.0, 'id': 3525, 'volume': 200, 'low': 50.0, 'open': 100.0},
                {'timestamp': datetime.datetime(2017, 1, 14, 14, 30,
                                                tzinfo=psycopg2.tz.FixedOffsetTimezone(offset=0, name=None)),
                 'symbol': 'MSFT', 'high': 50.0, 'close': 50.0, 'id': 3531, 'volume': 200, 'low': 50.0, 'open': 100.0},
                {'timestamp': datetime.datetime(2017, 1, 13, 14, 30,
                                                tzinfo=psycopg2.tz.FixedOffsetTimezone(offset=0, name=None)),
                 'symbol': 'AAPL', 'high': 50.0, 'close': 50.0, 'id': 3524, 'volume': 200, 'low': 50.0, 'open': 200.0},
                {'timestamp': datetime.datetime(2017, 1, 13, 14, 30,
                                                tzinfo=psycopg2.tz.FixedOffsetTimezone(offset=0, name=None)),
                 'symbol': 'MSFT', 'high': 50.0, 'close': 50.0, 'id': 3530, 'volume': 200, 'low': 50.0, 'open': 200.0},
                {'timestamp': datetime.datetime(2017, 1, 12, 14, 30,
                                                tzinfo=psycopg2.tz.FixedOffsetTimezone(offset=0, name=None)),
                 'symbol': 'AAPL', 'high': 50.0, 'close': 50.0, 'id': 3523, 'volume': 200, 'low': 50.0, 'open': 200.0},
                {'timestamp': datetime.datetime(2017, 1, 12, 14, 30,
                                                tzinfo=psycopg2.tz.FixedOffsetTimezone(offset=0, name=None)),
                 'symbol': 'MSFT', 'high': 50.0, 'close': 50.0, 'id': 3529, 'volume': 200, 'low': 50.0, 'open': 200.0},
                {'timestamp': datetime.datetime(2017, 1, 11, 14, 30,
                                                tzinfo=psycopg2.tz.FixedOffsetTimezone(offset=0, name=None)),
                 'symbol': 'AAPL', 'high': 50.0, 'close': 50.0, 'id': 3522, 'volume': 200, 'low': 50.0, 'open': 400.0},
                {'timestamp': datetime.datetime(2017, 1, 11, 14, 30,
                                                tzinfo=psycopg2.tz.FixedOffsetTimezone(offset=0, name=None)),
                 'symbol': 'MSFT', 'high': 50.0, 'close': 50.0, 'id': 3528, 'volume': 200, 'low': 50.0, 'open': 400.0},
                {'timestamp': datetime.datetime(2017, 1, 10, 14, 30,
                                                tzinfo=psycopg2.tz.FixedOffsetTimezone(offset=0, name=None)),
                 'symbol': 'AAPL', 'high': 50.0, 'close': 50.0, 'id': 3521, 'volume': 200, 'low': 50.0, 'open': 400.0},
                {'timestamp': datetime.datetime(2017, 1, 10, 14, 30,
                                                tzinfo=psycopg2.tz.FixedOffsetTimezone(offset=0, name=None)),
                 'symbol': 'MSFT', 'high': 50.0, 'close': 50.0, 'id': 3527, 'volume': 200, 'low': 50.0, 'open': 400.0}]

    def _query_adjustments(self, moment, interval):
        return [{'date': datetime.date(2050, 1, 1), 'symbol': 'AAPL', 'id': 8, 'dividend': 1.0, 'split': 1.0},
                {'date': datetime.date(2017, 1, 13), 'symbol': 'AAPL', 'id': 7, 'dividend': 1.0, 'split': 0.5},
                {'date': datetime.date(2017, 1, 12), 'symbol': 'MSFT', 'id': 9, 'dividend': 1.0, 'split': 0.5},
                {'date': datetime.date(2017, 1, 11), 'symbol': 'AAPL', 'id': 6, 'dividend': 1.0, 'split': 0.25}]

    def _query_values_for_symbols_and_time(self, feature, moment, symbol_list):
        return [{'timestamp': datetime.datetime(2017, 1, 11, 14, 30, tzinfo=psycopg2.tz.FixedOffsetTimezone(offset=0, name=None)),
                 'open': 400.0, 'symbol': 'MSFT'},
                {'timestamp': datetime.datetime(2017, 1, 11, 14, 30, tzinfo=psycopg2.tz.FixedOffsetTimezone(offset=0, name=None)),
                 'open': 400.0, 'symbol': 'AAPL'}]


@pytest.fixture
def sp500_data_source():
    data_source_config = {
        "database": "dbname=gabalese host=localhost",
        "exchange": "NYSE",
        "start": datetime.datetime(1999, 1, 1, tzinfo=pytz.utc),
        "end": datetime.datetime(1999, 3, 1, tzinfo=pytz.utc)
    }

    sp500_ds = MockSP500DataSource(data_source_config)
    return sp500_ds


def test_sp500_data_source_can_give_some_data(sp500_data_source):
    open_prices = sp500_data_source.values_for_symbols_feature_and_time(
        ['AAPL', 'MSFT'], 'open', datetime.datetime(2017, 1, 11, 14, 30))
    assert open_prices == [100, 200]


def test_sp500_source_can_get_data_for_datetime_and_interval(sp500_data_source):
    data = sp500_data_source.get_data(datetime.datetime(2017, 1, 15, 14, 30), datetime.timedelta(days=5))

    assert data['low']['AAPL'].loc[pd.Timestamp('2017-01-15 14:30:00')] == 50.0
    assert data['low']['AAPL'].loc[pd.Timestamp('2017-01-11 14:30:00')] == 12.5

    # adjusted opens should stay at 100
    assert data['open']['AAPL'].loc[pd.Timestamp('2017-01-15 14:30:00')] == 100.0
    assert data['open']['AAPL'].loc[pd.Timestamp('2017-01-11 14:30:00')] == 100.0
