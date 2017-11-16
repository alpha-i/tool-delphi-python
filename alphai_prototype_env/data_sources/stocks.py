from alphai_prototype_env.providers.hdf5 import HDF5DataProvider
from alphai_prototype_env.data_sources.base import AbstractDataSource


class StockLoader(AbstractDataSource):

    def __init__(self, config):
        super().__init__(config)
        self.filename = config.filename
        self.exchange = config.exchange
        self.hdf5_data_provider = HDF5DataProvider(self.filename, self.exchange)

    def get_train_data(self):

        data = self.hdf5_data_provider.get_data(self._train_start, self._train_end)

        return data

    def get_test_data(self):

        data = self.hdf5_data_provider.get_data(self._test_start, self._test_end)

        return data
