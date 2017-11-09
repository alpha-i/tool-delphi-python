from alphai_prototype_env.data_providers.hdf5 import HDF5DataProvider


class StockLoader(object):

    def __init__(self, config):
        self.filename = config.filename
        self.exchange = config.exchange
        self.train_start = config.train_start
        self.train_end = config.train_end
        self.test_start = config.test_start
        self.test_end = config.test_end
        self.data_provider = HDF5DataProvider(self.filename, self.exchange)

    def get_train_data(self):

        data = self.data_provider.get_data(self.train_start, self.train_end)

        return data

    def get_test_data(self):

        data = self.data_provider.get_data(self.test_start, self.test_end)

        return data
