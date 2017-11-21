from abc import ABCMeta


class OracleConfiguration(metaclass=ABCMeta):
    # TODO: make me a marshmallow schema

    def __init__(self, training_frequency, prediction_frequency, prediction_offset, prediction_horizon):
        self.prediction_horizon = prediction_horizon
        self.prediction_offset = prediction_offset
        self.prediction_frequency = prediction_frequency
        self.training_frequency = training_frequency
