import numpy as np


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

    def compute_metrics(self, predictions, test_data):

        return None
