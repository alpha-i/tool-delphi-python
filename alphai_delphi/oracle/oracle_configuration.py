from alphai_delphi.configuration.schemas import SchedulingConfigurationSchema, AttributeDict


class OracleConfiguration:

    SCHEDULING_SCHEMA = SchedulingConfigurationSchema()

    def __init__(self, config):
        """
        Loads a dict containing the configuration: Example.
        {
            'scheduling': {
                'prediction_frequency':
                    {
                        'frequency_type': 'DAILY', # frequency type
                        'days_offset': 0, # days offset from the beginning of the week
                        'minutes_offset': 15 # minutes offset from the market opening
                    },
                'training_frequency':
                    {
                        'frequency_type': 'WEEKLY', # frequency type
                        'days_offset': 0, # days offset from the beginning of the week
                        'minutes_offset': 15 # minutes offset from the market opening
                    },
            },
            'oracle': {
                # oracle dependent parameters
            }
        }
        :param config:
        :type config: dict
        """
        result = self.SCHEDULING_SCHEMA.load(config['scheduling'])
        if result.errors:
            raise Exception(result.errors)

        self.scheduling = result.data
        self.oracle = AttributeDict(config['oracle'])
