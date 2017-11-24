from marshmallow import Schema, fields
from marshmallow_enum import EnumField

from delphi.scheduler.abstract_scheduler import SchedulingFrequencyType


class AttributeDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class BaseSchema(Schema):
    @property
    def dict_class(self):
        return AttributeDict


class SchedulingFrequencySchema(BaseSchema):
    frequency_type = EnumField(SchedulingFrequencyType, required=True)
    minutes_offset = fields.Integer(default=0)
    days_offset = fields.Integer(default=0)


class OracleSchedulingConfigurationSchema(BaseSchema):
    prediction_horizon = fields.TimeDelta(precision='hours', required=True)
    prediction_frequency = fields.Nested(SchedulingFrequencySchema(), required=True)
    prediction_delta = fields.TimeDelta(precision='hours', required=True)

    training_frequency = fields.Nested(SchedulingFrequencySchema(), required=True)
    training_delta = fields.TimeDelta(precision='hours', required=True)


class ControllerConfigurationSchema(BaseSchema):
    start_date = fields.DateTime(required=True)
    end_date = fields.DateTime(required=True)
    performance_result_output = fields.Str(required=True)
