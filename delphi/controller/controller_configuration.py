from abc import ABCMeta


class ControllerConfiguration(metaclass=ABCMeta):
    # TODO: make me a marshmallow schema
    
    def __init__(self, start_date, end_date):
        """
        :param start_date: Start of the backtest / run
        :param end_date: End of the backtest / run
        """
        self.end_date = end_date
        self.start_date = start_date
