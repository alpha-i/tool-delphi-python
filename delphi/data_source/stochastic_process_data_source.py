import pandas as pd
from delphi.data_source.abstract_data_source import AbstractDataSource
from alphai_data_sources.data_sources import DataSourceGenerator
from alphai_data_sources.generator import BatchGenerator, BatchOptions


class StochasticProcessDataSource(AbstractDataSource):
    def __init__(self, configuration):
        super().__init__(configuration)
        self._data_dict = {}
        self._setup_data()

    def _setup_data(self):
        time_index = pd.date_range(start=self.start(), end=self.end(), freq='min')
        print(len(time_index))
        data_source_generator = DataSourceGenerator()
        data_source = data_source_generator.make_data_source(self.config["series_name"])
        batch_number = 0
        batch_size = len(time_index)
        for_training = True
        dtype = 'float32'
        batch_options = BatchOptions(batch_size, batch_number, for_training, dtype)
        batch_generator = BatchGenerator()
        features, labels = batch_generator.get_batch(batch_options, data_source)
        print(len(features))


    def start(self):
        return self.config["start"]

    def end(self):
        return self.config["end"]

    def get_data(self, current_datetime, interval):
        return self._data_dict

    def values_for_symbols_feature_and_time(self, symbol_list, feature, current_datetime):
        pass


if __name__ == '__main__':
    configuration = \
        {
            'start': '20100101',
            'end': '20100131',
            'series_name': 'stochastic_walk'
        }

    data_source = StochasticProcessDataSource(configuration)
