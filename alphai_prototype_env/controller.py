from alphai_prototype_env.metrics import Metrics


class Controller(object):

    def __init__(self, baseline):
        self.baseline = baseline
        self.metrics = Metrics()

    def train_model(self, model, data_source):

        model.reset()
        train_data = data_source.get_train_data()
        model.train(train_data)
        model.save()

    def test_model(self, model, data_source):

        model.load()

        prediction_schedule = self.make_schedule(model, data_source)
        # prediction_schedule = [{"prediction_vars":{"close":['APPL]},
        #                         "prediction_datetime: '20010120',
        #                         "last_datetime": '20010119' }]

        predictions = []
        actuals = []

        for prediction_event in prediction_schedule:
            query = model.generate_query(prediction_event)
            # query = {"t_start":'20000101', "t_end": '20010119', "feature_vars": {close:['APPL']}}
            sample = data_source.get_sample(query, prediction_event)  # check not asking for data after last_datetime
            # sample = {"close":pd.DataFrame}
            prediction = model.make_prediction(sample)
            # prediction = {"mean": pd.Series, "covariance": pd.DataFrame}
            actual = data_source.get_actual(prediction_event)
            # actual = {"mean": pd.Series, "covariance": pd.DataFrame}

            predictions.append(prediction)
            actuals.append(actual)

        model_metrics = self.metrics.compute_metrics(predictions, actuals)

        return model_metrics

    def compare_to_baseline(self, model, data_source):

        self.check_model_compatible_with_baseline(model)

        self.train_model(self.baseline, data_source)
        self.train_model(model, data_source)

        baseline_metrics = self.test_model(self.baseline, data_source)
        model_metrics = self.test_model(model, data_source)

        results = {"baseline_metrics": baseline_metrics, "model_metrics": model_metrics}

        return results

    def check_model_compatible_with_baseline(self, model):

        if model.frequency != self.baseline.frequency:
            raise ValueError('model prediction frequency should be equal to the baseline prediction frequency')

        if model.delta != self.baseline.delta:
            raise ValueError('model prediction delta should be equal to the baseline prediction delta')

        if model.offset != self.baseline.delta:
            raise ValueError('model prediction offset should be equal to the baseline prediction offset')

    def compare_on_multiple_data_sources(self, model, data_source_list):

        results_dict = {}
        for data_source in data_source_list:
            results = self.compare_to_baseline(model, data_source)
            results_dict[data_source.__class__.__name__] = results
