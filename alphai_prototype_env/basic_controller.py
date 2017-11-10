from collections import namedtuple

from alphai_prototype_env.metrics import Metrics

DataSlice = namedtuple('Slice', 'features targets')

PREDICT = "predict"
RETRAIN = "retrain"


class BasicController(object):

    def __init__(self):
        self.metrics = Metrics()

    def train_oracle(self, oracle, data_source):

        oracle.reset()
        train_data = data_source.get_train_data()
        oracle.train(train_data)
        oracle.save()

    def test_oracle(self, oracle, data_source):

        oracle.load()

        test_data = data_source.get_test_data()
        schedule = self.get_schedule(oracle, data_source)

        prediction_list = []
        actual_list = []

        for event in schedule:
            data_slice = self.get_data_slice(event, oracle, test_data)

            if event.type == PREDICT:
                prediction = oracle.predict(data_slice)
                prediction_list.append(prediction)

                actual = self.get_actual(event, test_data)
                actual_list.append(actual)

            elif event.type == RETRAIN:
                oracle.train(data_slice)

        model_metrics = self.metrics.compute_metrics(prediction_list, actual_list)

        return model_metrics

    def get_data_slice(self, event, oracle, test_data):

        query = oracle.generate_query(event)
        self.check_query(query, event)
        data_slice = self.slice_test_data(test_data, query)

        return data_slice

    def get_schedule(self, oracle, data_source):

        schedule = []

        return schedule

    def slice_test_data(self, test_data, query):

        data_slice = DataSlice

        return data_slice

    def check_query(self, query, event):

        return None

    def get_actual(self, event, test_data):

        actual = {}

        return actual



