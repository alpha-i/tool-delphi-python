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

    def run_oracle(self, oracle, data_source, mode):

        oracle.load()
        data = data_source.get_data(mode)
        start, end = data_source.get_start_end_datetimes(mode)
        schedule = self.get_schedule(oracle, start, end)

        prediction_list = []
        actual_list = []

        for event in schedule:
            query = oracle.generate_query(event)

            if query.end > event.timestamp - oracle.prediction_delta:
                raise ValueError('Not allowed to access data in the future.')

            data_window = data_source.get_data_window(data, query.start, query.end)

            if event.type == PREDICT:
                prediction = oracle.predict(data_window)
                prediction_list.append(prediction)

                actual = data_source.get_data_window(data, event.prediction_start, event.prediction_end)
                actual_list.append(actual)

            elif event.type == RETRAIN:
                oracle.train(data_window)

        model_metrics = self.metrics.compute_metrics(prediction_list, actual_list)

        return model_metrics

    def get_schedule(self, oracle, start, end):

        schedule = []

        return schedule
