from abc import ABCMeta, abstractmethod


class AbstractController(metaclass=ABCMeta):
    def __init__(self, configuration, oracle, datasource):
        self.configuration = configuration
        self.oracle = oracle
        self.datasource = datasource

    @abstractmethod
    def run(self):
        raise NotImplementedError
