import numpy as np
import pandas as pd

from alphai_prototype_env.oracles.base import AbstractOracle


class ConstantOracle(AbstractOracle):

    def reset(self):
        pass

    def save(self, filepath):
        pass

    def load(self, filepath):
        pass

    def transform(self, data):
        pass

    def train(self, train_data):
        pass

    def predict(self, predict_data):

        num_stocks = len(predict_data['close'].columns)
        mean = predict_data['close'].iloc[-1]
        covariance = pd.DataFrame(data=np.eye(num_stocks), index=mean.index, columns=mean.index)

        return mean, covariance
