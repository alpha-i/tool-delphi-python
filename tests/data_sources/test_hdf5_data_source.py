import datetime
import os
from unittest import TestCase

import numpy as np

from delphi.data_source.hdf5_data_source import HDF5DataSource


class TestHDF5DataProvider(TestCase):

    @classmethod
    def setUpClass(cls):

        filename = os.path.join(os.path.dirname(__file__), '..', 'resources', '19990101_19990301_3_stocks.hdf5')

        cls.config = {
            "filename": filename,
            "exchange": "NYSE",
            "start": datetime.datetime(1999, 1, 1),
            "end": datetime.datetime(1999, 3, 1)
        }

        cls.data_source = HDF5DataSource(cls.config)

    def test_start(self):

        assert self.config["start"] == self.data_source.start()

    def test_end(self):

        assert self.config["end"] == self.data_source.end()

    def test_get_data(self):

        expected_sybols = ['AMZN', 'BAX', 'CSCO']

        current_datetime = datetime.datetime(1999, 1, 15, 14, 30)
        interval = datetime.timedelta(days=4)

        data_dict = self.data_source.get_data(current_datetime, interval)

        assert data_dict['close'].iloc[0].name.day == 11
        assert data_dict['close'].iloc[-1].name.day == 15

        assert np.all([data_dict['close'].columns == expected_sybols])
