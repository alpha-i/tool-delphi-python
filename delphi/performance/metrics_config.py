from abc import ABCMeta


class MetricsConfiguration(metaclass=ABCMeta):
    def __init__(self, required_metrics):
        self.required_metrics = required_metrics
