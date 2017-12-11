import datetime
from collections import defaultdict

import psycopg2
from psycopg2.extras import DictCursor

from alphai_delphi.data_source import AbstractDataSource
import pandas as pd


class SP500DataSource(AbstractDataSource):
    def __init__(self, configuration):
        super().__init__(configuration)
        self.dsn = self.config['database']
        self.connection = psycopg2.connect(self.dsn, cursor_factory=DictCursor)

    @property
    def end(self):
        return self.config['end']

    @property
    def start(self):
        return self.config['start']

    def _query_minutes(self, moment, interval):
        with self.connection.cursor() as cursor:
            minutes_query = """
              SELECT * FROM minutes WHERE timestamp BETWEEN %s and %s ORDER BY timestamp DESC
            """
            cursor.execute(minutes_query, (moment - interval, moment))
            minutes_data = cursor.fetchall()
        return minutes_data

    def _query_adjustments(self, moment, interval):
        with self.connection.cursor() as cursor:
            adjustments_query = """
              SELECT * FROM adjustments WHERE date >= %s ORDER BY date DESC
            """
            cursor.execute(adjustments_query, (moment - interval,))
            adjustments_data = cursor.fetchall()
        return adjustments_data

    def _query_values_for_symbols_and_time(self, feature, moment, symbol_list):
        with self.connection.cursor() as cursor:
            query = """
            SELECT symbol, timestamp, {} FROM minutes WHERE timestamp = %s AND symbol IN %s
            """.format(feature)  # stupid psycopg2 doesn't parametrize column names
            cursor.execute(query, (moment, tuple(symbol_list)))
            values = cursor.fetchall()
        return values

    def values_for_symbols_feature_and_time(self, symbol_list, feature, current_datetime):
        values = self._query_values_for_symbols_and_time(feature, current_datetime, symbol_list)
        for value in values:
            symbol = value['symbol']
            timestamp = value['timestamp']
            adjustment = self._get_adjustment_for_symbol_and_timestamp(
                symbol, timestamp, self._query_adjustments(current_datetime, datetime.timedelta(days=1)))
            factor = adjustment['split'] * adjustment['dividend']
            value[feature] *= factor
        symbols_and_bar = {v['symbol']: v[feature] for v in values}
        return [symbols_and_bar[symbol] for symbol in symbol_list]

    def _get_adjustment_for_symbol_and_timestamp(self, symbol, timestamp, adjustments_data):
        date = timestamp.date()
        applicable_adjustments = list(filter(lambda x: x['symbol'] == symbol and x['date'] >= date, adjustments_data))
        if applicable_adjustments:
            return applicable_adjustments[-1]
        return None

    def get_data(self, current_datetime, interval):
        minutes_data = self._query_minutes(current_datetime, interval)
        adjustments_data = self._query_adjustments(current_datetime, interval)

        for minute in minutes_data:
            adjustment = self._get_adjustment_for_symbol_and_timestamp(
                minute['symbol'], minute['timestamp'], adjustments_data)
            if adjustment:
                factor = adjustment['split'] * adjustment['dividend']
                minute['open'] *= factor
                minute['high'] *= factor
                minute['low'] *= factor
                minute['close'] *= factor

        aggregated_data = defaultdict(list)
        for minute in minutes_data:
            for key in ['low', 'volume', 'high', 'open', 'close']:
                if not aggregated_data[key]:
                    aggregated_data[key].append(
                        {'timestamp': minute['timestamp'], minute['symbol']: minute[key]}
                    )
                else:
                    if aggregated_data[key][-1]['timestamp'] == minute['timestamp']:
                        aggregated_data[key][-1][minute['symbol']] = minute[key]
                    else:
                        aggregated_data[key].append(
                            {'timestamp': minute['timestamp'], minute['symbol']: minute[key]}
                        )

        return {key: pd.DataFrame(aggregated_data[key]).set_index('timestamp') for key in aggregated_data}
