from enum import Enum


class DefaultMetrics(Enum):

    mean_vector = 'mean_vector'
    covariance_matrix = 'covariance_matrix'
    initial_values = 'initial_values'
    final_values = 'final_values'
    returns_actuals = 'returns_actuals' #log-return of initial and final values

    @staticmethod
    def get_metrics():
        return [member.value for member in list(DefaultMetrics.__members__.values())]
