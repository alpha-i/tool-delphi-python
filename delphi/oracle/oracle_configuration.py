from delphi.configuration.schemas import OracleSchedulingConfigurationSchema


class OracleConfiguration:

    SCHEMA = OracleSchedulingConfigurationSchema()

    def __init__(self, config):
        """
        Loads a dict containing the configuration
        {
        'scheduling': {
            'prediction_horizon': '',
            'prediction_offset': '',
            'prediction_frequency': '',
            'prediction_delta': '',

            'training_frequency': '',
            'training_delta': '',
        },
        'oracle': {
            }
        }
        :param config:
        :type config: dict
        """
        result = self.SCHEMA.load(config['scheduling'])
        if result.errors:
            raise Exception(result.errors)

        self.scheduling = result.data
        self.oracle = config['oracle']

    def get(self, field):
        return self.oracle.get(field)
