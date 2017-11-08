import numpy as np


class Controller(object):

    def __init__(self, baseline):
        self.baseline = baseline
        self.metrics = Metrics()

    def train_model(self, model, data_source):

        model.reset()
        model.train(data_source.train_data)
        model.save()

    def test_model(self, model, data_source):

        model.load()
        predictions = model.predict(data_source.test_data)   # predictions should be a dict of data_frames
        model_metrics = self.metrics.compute_metrics(predictions, data_source.test_data)

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

    def multiple_compare_to_baseline(self, model, data_source_list):

        results_dict = {}
        for data_source in data_source_list:
            results = self.compare_to_baseline(model, data_source)
            results_dict[data_source.name] = results


class Metrics(object):

    def compute_log_likelihood(self, x, mu, sigma):
        """
        Computes the log likelihood of a data point x for a multivariate Gaussian
        :param x: np.array([1, n])
        :param mu: np.array([1, n])
        :param sigma: np.array([n,n])
        :return: log_likelihood: float
        """

        residuals = x - mu
        sigma_inv = np.linalg.inv(sigma)
        quadratic = np.dot(residuals, np.dot(sigma_inv, residuals.T))
        n = len(residuals)
        constant = n * np.log10(2 * np.pi)
        log_det = np.log10(np.linalg.det(sigma))

        log_likelihood = -0.5 * (quadratic + log_det + constant)

        return log_likelihood

    def compute_binary_accuracy(self):

        return None

    def compute_mean_log_likelihood(self, predictions, test_data):

        return None

    def compute_metrics(self, predictions, test_data):

        return None
