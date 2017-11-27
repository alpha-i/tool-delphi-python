import datetime

import numpy as np
import pandas as pd

from delphi.oracle.abstract_oracle import AbstractOracle, PredictionResult, OracleAction


class ConstantOracle(AbstractOracle):

    @property
    def target_feature(self):
        return self.config.target_feature

    def save(self):
        pass

    def load(self):
        pass

    def train(self, data):
        pass

    def predict(self, data, timestamp):

        constant_variance = self.config["constant_variance"]

        symbols = data['close'].columns
        num_symbols = len(symbols)

        mean = data['close'].iloc[-1]
        covariance = pd.DataFrame(data=constant_variance * np.eye(num_symbols), index=symbols, columns=symbols)

        prediction = PredictionResult(mean, covariance, timestamp)

        return prediction

    def get_delta_for_event(self, event):

        if event == OracleAction.TRAIN:
            interval = datetime.timedelta(days=1)
        elif event == OracleAction.PREDICT:
            interval = datetime.timedelta(days=7)
        else:
            raise ValueError('Unexpected scheduling event type: {}'.format(event))

        return interval
