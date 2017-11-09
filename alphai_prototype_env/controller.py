from alphai_prototype_env.metrics import Metrics


class Controller(object):

    def __init__(self, baseline):
        self.baseline = baseline
        self.metrics = Metrics()

    def train_model(self, oracle, data_source):

        oracle.reset()
        train_data = data_source.get_train_data()
        oracle.train(train_data)
        oracle.save()

    def retrain_model(self, oracle, retrain_data):

        oracle.load()
        oracle.train(retrain_data)
        oracle.save()

    def test_model(self, oracle, data_source):

        oracle.load()

        schedule = self.make_schedule(oracle, data_source)
        # schedule = [{"type": "predict", "prediction_datetime: '20010120', "last_datetime": '20010119' }]

        predictions = []
        actuals = []

        for event in schedule:
            if event["type"] == "predict":
                query = oracle.generate_query(event)

                predict_data = self.get_data(query, data_source, event)
                prediction = oracle.make_prediction(predict_data)
                actual = self.get_actual(data_source, event)

                predictions.append(prediction)
                actuals.append(actual)

            elif event["type"] == "retrain":
                query = oracle.generate_query(event)
                retrain_data = self.get_data(query, data_source, event)
                self.retrain_model(oracle, retrain_data)

        model_metrics = self.metrics.compute_metrics(predictions, actuals)

        return model_metrics

    def compare_to_baseline(self, oracle, data_source):

        self.check_model_compatible_with_baseline(oracle)

        self.train_model(self.baseline, data_source)
        self.train_model(oracle, data_source)

        baseline_metrics = self.test_model(self.baseline, data_source)
        model_metrics = self.test_model(oracle, data_source)

        results = {"baseline_metrics": baseline_metrics, "model_metrics": model_metrics}

        return results

    def check_model_compatible_with_baseline(self, oracle):

        if oracle.frequency != self.baseline.frequency:
            raise ValueError('model prediction frequency should be equal to the baseline prediction frequency')

        if oracle.delta != self.baseline.delta:
            raise ValueError('model prediction delta should be equal to the baseline prediction delta')

        if oracle.offset != self.baseline.delta:
            raise ValueError('model prediction offset should be equal to the baseline prediction offset')

    def compare_multiple_oracles(self, oracle_list, data_source):

        results_dict = {}
        for oracle in oracle_list:
            results = self.compare_to_baseline(oracle, data_source)
            results_dict[oracle.__class__.__name__] = results
