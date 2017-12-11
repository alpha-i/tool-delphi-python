from alphai_delphi.configuration.schemas import OracleSchedulingConfigurationSchema, AttributeDict


class OracleConfiguration:

    SCHEMA = OracleSchedulingConfigurationSchema()

    def __init__(self, config):
        """
        Loads a dict containing the configuration: Example.
        {
            'scheduling': {
                'prediction_horizon': 240, # how many HOURS in the future you want to predict
                'prediction_frequency':
                    {
                        'frequency_type': 'DAILY', # frequency type
                        'days_offset': 0, # days offset from the beginning of the week
                        'minutes_offset': 15 # minutes offset from the market opening
                    },
                'prediction_delta': 10, # how many DAYS of data are needed for the prediction
            
                'training_frequency':
                    {
                        'frequency_type': 'WEEKLY', # frequency type
                        'days_offset': 0, # days offset from the beginning of the week
                        'minutes_offset': 15 # minutes offset from the market opening
                    },
                'training_delta': 20, # how many DAYS of data are needed for the training
            },
            'oracle': {
                # oracle dependent parameters
            }
        }
        :param config:
        :type config: dict
        """
        result = self.SCHEMA.load(config['scheduling'])
        if result.errors:
            raise Exception(result.errors)

        self.scheduling = result.data
        self.oracle = AttributeDict(config['oracle'])
