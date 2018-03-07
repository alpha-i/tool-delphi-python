from enum import Enum
from marshmallow import Schema, fields
from marshmallow_enum import EnumField

from alphai_delphi.scheduler.abstract_scheduler import SchedulingFrequencyType


class PredictionHorizonUnit(Enum):

    days = 'days'
    seconds = 'seconds'
    microseconds = 'microseconds'
    milliseconds = 'milliseconds'
    minutes = 'minutes'
    hours = 'hours'
    weeks = 'weeks'


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


class SchedulingConfigurationSchema(BaseSchema):
    """
    prediction_horizon: how many HOURS in the future you want to predict

    prediction_frequency: The object which defines the scheduling
    prediction_delta: how many DAYS of data are needed for the prediction

    training_frequency:the type of frequency for training
    training_delta: how many DAYS of data are needed for the training
    """

    prediction_frequency = fields.Nested(SchedulingFrequencySchema(), required=True)
    training_frequency = fields.Nested(SchedulingFrequencySchema(), required=True)


class PredictionHorizonConfigurationSchema(BaseSchema):

    unit = EnumField(PredictionHorizonUnit, required=True)
    value = fields.Integer()


class DataTransformationConfigurationSchema(BaseSchema):

    feature_config_list = fields.List(fields.Dict)
    features_ndays = fields.Integer()
    features_resample_minutes = fields.Integer()
    fill_limit = fields.Integer()


class OracleConfigurationSchema(BaseSchema):
    prediction_delta = fields.Integer()
    training_delta = fields.Integer()

    prediction_horizon = fields.Nested(PredictionHorizonConfigurationSchema)
    data_transformation = fields.Nested(DataTransformationConfigurationSchema)

    model = fields.Dict()
    universe = fields.Dict(required=False, allow_none=True)

class ControllerConfigurationSchema(BaseSchema):
    """
    start_date: the start date of the simulation
    end_date : the end date of the simulation
    """
    start_date = fields.DateTime(required=True)
    end_date = fields.DateTime(required=True)
