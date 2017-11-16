import os
from unittest import TestCase
from datetime import datetime

import numpy as np

from alphai_prototype_env.providers.hdf5 import HDF5DataProvider


class TestHDF5DataProvider(TestCase):

    @classmethod
    def setUpClass(cls):

        data_file = os.path.join(os.path.dirname(__file__), '..', 'resources', '19990101_19990301_3_stocks.hdf5')
        exchange = 'NYSE'

        cls.data_provider = HDF5DataProvider(data_file, exchange)

    def test_get_data(self):

        start_date = datetime(1999, 1, 11, 14, 30)
        end_date = datetime(1999, 1, 15, 14, 30)
        index = ['AMZN', 'BAX', 'CSCO']

        data_dict = self.data_provider.get_data(start_date, end_date)

        assert data_dict['close'].iloc[0].name.day == 11
        assert data_dict['close'].iloc[-1].name.day == 15

        assert np.all([data_dict['close'].columns == index])
