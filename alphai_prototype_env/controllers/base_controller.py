from alphai_prototype_env.metrics.metrics import Metrics

PREDICT = "predict"
RETRAIN = "retrain"


class BaseController(object):

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
            query = oracle.generate_query(event)

            if query.end > event.timestamp - oracle.prediction_delta:
                raise ValueError

            data_window = data_source.get_test_data_window(test_data, query.start, query.end)

            if event.type == PREDICT:
                prediction = oracle.predict(data_window)
                prediction_list.append(prediction)

                actual = data_source.get_test_actual(event.timestamp)
                actual_list.append(actual)

            elif event.type == RETRAIN:
                oracle.train(data_window)

        model_metrics = self.metrics.compute_metrics(prediction_list, actual_list)

        return model_metrics

    def get_schedule(self, oracle, data_source):

        schedule = []

        return schedule


