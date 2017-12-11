from abc import ABCMeta


class DataSourceConfiguration(metaclass=ABCMeta):

    def __init__(self, start, end):
        self.start = start
        self.end = end
