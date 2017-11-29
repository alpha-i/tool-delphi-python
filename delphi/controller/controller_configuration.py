from abc import ABCMeta

from delphi.configuration.schemas import ControllerConfigurationSchema


class ControllerConfiguration(metaclass=ABCMeta):

    SCHEMA = ControllerConfigurationSchema()

    def __init__(self, configuration):
        """

        :param configuration: Dictionary of configuration
        :type configuration: dict

        """
        result, error = self.SCHEMA.load(configuration)
        if result.errors:
            raise Exception(result.errors)

        self.end_date = result.start_date
        self.start_date = result.end_date
