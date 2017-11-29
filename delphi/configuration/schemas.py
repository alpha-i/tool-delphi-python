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
    """
    frequency_type: the type of frequency
    minutes_offset: minutes offset from the market opening
    days_offset : days offset from the beginning of the week
    """
    frequency_type = EnumField(SchedulingFrequencyType, required=True)
    minutes_offset = fields.Integer(default=0)
    days_offset = fields.Integer(default=0)


class OracleSchedulingConfigurationSchema(BaseSchema):
    """
    prediction_horizon: how many HOURS in the future you want to predict

    prediction_frequency: The object which defines the scheduling
    prediction_delta: how many DAYS of data are needed for the prediction

    training_frequency:the type of frequency for training
    training_delta: how many DAYS of data are needed for the training
    """
    prediction_horizon = fields.TimeDelta(precision='hours', required=True)

    prediction_frequency = fields.Nested(SchedulingFrequencySchema(), required=True)
    prediction_delta = fields.TimeDelta(precision='days', required=True)

    training_frequency = fields.Nested(SchedulingFrequencySchema(), required=True)
    training_delta = fields.TimeDelta(precision='days', required=True)


class ControllerConfigurationSchema(BaseSchema):
    """
    start_date: the start date of the simulation
    end_date : the end date of the simulation
    """
    start_date = fields.DateTime(required=True)
    end_date = fields.DateTime(required=True)
