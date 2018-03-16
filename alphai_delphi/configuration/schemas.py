from enum import Enum

from alphai_delphi.scheduler.abstract_scheduler import SchedulingFrequencyType
from marshmallow import Schema, fields
from marshmallow_enum import EnumField


class TimeDeltaUnit(Enum):
    days = 'days'
    seconds = 'seconds'
    microseconds = 'microseconds'
    milliseconds = 'milliseconds'
    minutes = 'minutes'
    hours = 'hours'
    weeks = 'weeks'


class BaseSchema(Schema):
    pass


class SchedulingFrequencySchema(BaseSchema):
    """
    frequency_type: the type of frequency
    minutes_offset: minutes offset from the opening
    days_offset : days offset from the beginning of the week (0 for Monday, 1 for Tuesday...)
    """
    frequency_type = EnumField(SchedulingFrequencyType, required=True)
    minutes_offset = fields.Integer(default=0, missing=0)
    days_offset = fields.Integer(default=0, missing=0)


class SchedulingConfigurationSchema(BaseSchema):
    prediction_frequency = fields.Nested(SchedulingFrequencySchema, required=True)
    training_frequency = fields.Nested(SchedulingFrequencySchema, required=True)


class TimeDeltaConfigurationSchema(BaseSchema):
    unit = EnumField(TimeDeltaUnit, required=True)
    value = fields.Integer()


class OracleConfigurationSchema(BaseSchema):
    prediction_delta = fields.Nested(TimeDeltaConfigurationSchema, required=True)
    training_delta = fields.Nested(TimeDeltaConfigurationSchema, required=True)

    prediction_horizon = fields.Nested(TimeDeltaConfigurationSchema, required=True)
    data_transformation = fields.Dict(required=True)

    model = fields.Dict(required=True)
    universe = fields.Dict(required=False, allow_none=True)


class ControllerConfigurationSchema(BaseSchema):
    """
    start_date: the start date of the simulation
    end_date : the end date of the simulation
    """
    start_date = fields.Date(required=True)
    end_date = fields.Date(required=True)


class InvalidConfiguration(Exception):
    pass
