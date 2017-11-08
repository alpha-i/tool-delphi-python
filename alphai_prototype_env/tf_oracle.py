import numpy as np
import pandas as pd
import tensorflow as tf

from alphai_prototype_env.abstract_oracle import AbstractOracle
from alphai_finance.data.cleaning import resample_ohlcv


class TFOracle(AbstractOracle):
    """
    An example of a tensorflow based oracle that takes the data, resamples it to 15 min frequency, log transforms it,
    then makes features out of close and volume for ['APPL', 'MSFT']
    and uses the past day of data to try to predict the close in an hours time
    """

    def __init__(self, frequency, delta, offset):
        super().__init__(frequency, delta, offset)
        self.batch_size = 24
        self.num_epochs = 10
        self.num_inputs = 10
        self.num_outputs = 5
        self.resample_rate = '15T'
        self.chosen_symbols = ['APPL', 'MSFT']
        self.chosen_vars = ['Close', 'Volume']

    def transform(self, data):

        data = resample_ohlcv(data, self.resample_rate)
        num_steps = 0
        for key in self.chosen_vars:
            num_steps = max(num_steps, len(data[key].index))

        transformed_data = np.zeros([num_steps, len(self.chosen_symbols) * len(self.chosen_vars)])

        for i, key in enumerate(self.chosen_vars):
            transformed_data[i:i + len(self.chosen_symbols)] = data[key]









    def reset(self):

        tf.reset_default_graph()

        self.x_ph = tf.placeholder(tf.float32, shape=[None, self.num_inputs])
        self.y_ph = tf.placeholder(tf.float32, shape=[None, self.num_outputs])

        W = tf.Variable(tf.random_normal(shape=[self.num_inputs, self.num_outputs]))
        b = tf.Variable(tf.random_normal(shape=[None, self.num_outputs]))

        self.output = tf.nn.relu(tf.matmul(self.x_ph, W) + b)

        self.loss = tf.reduce_mean(tf.square(self.output - self.y_ph))

        self.train_step = tf.train.AdamOptimizer.minimize(loss=self.loss)

    def train(self, train_data):

        transformed_train_data = self.transform(train_data)

        with tf.Session as sess:
            for epoch in range(self.num_epochs):
                features, labels = self.get_batch(transformed_train_data)
                sess.run(self.train_step, feed_dict={self.x_ph: features, self.y_ph: labels})

    def predict(self, predict_data):

        transformed_predict_data = self.train(predict_data)
        [num_steps, num_vars] = transformed_predict_data.shape

        predictions = np.zeros([num_steps, num_vars])

        with tf.Session as sess:
            for step in range(num_steps):
                features, _ = self.get_next(transformed_predict_data, step)
                output = sess.run(self.y_hat, feed_dict={self.x_ph: features})
                predictions[step, :] = output

        predictions_dict = {"close": pd.DataFrame(data=predictions, columns=predict_data.columns)}

        return predictions_dict









