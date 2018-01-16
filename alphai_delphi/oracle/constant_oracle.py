import datetime

import numpy as np
import pandas as pd

from alphai_delphi.oracle.abstract_oracle import AbstractOracle, PredictionResult, OracleAction


class ConstantOracle(AbstractOracle):

    def _sanity_check(self):
        return True

    def resample(self, data):
        return data

    def fill_nan(self, data):
        return data

    def global_transform(self, data):
        return data

    def get_universe(self, data):
        return pd.DataFrame()

    @property
    def target_feature(self):
        return self.config.target_feature

    def save(self):
        pass

    def load(self):
        pass

    def train(self, data, current_timestamp):
        pass

    def predict(self, data, current_timestamp, target_timestamp):

        # constant_variance = self.config["constant_variance"]
        #
        # close_df = data.sel(raw_features='close').to_dataframe().drop(labels=["raw_features"], axis=1)
        #
        # symbols = close_df.columns
        # num_symbols = len(symbols)
        #
        # mean = close_df.iloc[-1]
        # mean.name = mean.name.tz_localize('America/New_York').astimezone('UTC')
        #
        # covariance = pd.DataFrame(data=constant_variance * np.eye(num_symbols), index=symbols, columns=symbols)
        #
        # prediction = PredictionResult(mean, covariance, current_timestamp, target_timestamp)
        #
        # return prediction

        constant_variance = self.config["constant_variance"]

        symbols = data['close'].columns
        num_symbols = len(symbols)

        mean = data['close'].iloc[-1]
        covariance = pd.DataFrame(data=constant_variance * np.eye(num_symbols), index=symbols, columns=symbols)

        prediction = PredictionResult(mean, covariance, current_timestamp, target_timestamp)

        return prediction

    def get_delta_for_event(self, event):

        if event == OracleAction.TRAIN:
            interval = datetime.timedelta(days=1)
        elif event == OracleAction.PREDICT:
            interval = datetime.timedelta(days=7)
        else:
            raise ValueError('Unexpected scheduling event type: {}'.format(event))

        return interval
