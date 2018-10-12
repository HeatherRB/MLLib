
import numpy as np
from ml.utils import *


class _LinearClassifier:
    """Base class for a linear classifier

    Attributes:
        w - array of weights
    """

    def __init__(self):

        # initiate the array of weights
        self.w = None

    def _net_input(self, x):
        """ Function to calculate the net input z = x.w

        Input:
            x - array of training data (without first column of 1s)
        Returns:
            z = x.w[1:] + w[0]
        """

        n_samples = x.shape[0]
        x_ext = np.concatenate((np.ones((n_samples, 1)), x), axis=1)

        return np.dot(x_ext, self.w)

    def _activation(self, x):
        """ Default activation function (perceptron)
        Inputs:
            a - array
        Outputs:
            phi(a) - value of perceptron activation function
        """

        return np.where(x >= 0, 1, -1)

    def predict(self, x):
        """ Function to predict the class of each sample (row) in x

        Inputs:
            x - training data (without first column of 1s)
        Returns:
            array of predicted classifications (values +1/-1)

        For step function, predicted class of each sample v is:
            +1  if v.w >= 0
            -1  otherwise
        """

        return np.where(self._activation(self._net_input(x)) >= 0, 1, -1)

    def _delta_w(self, x, y):
        """ Function to calculate the (unscaled) change to the weights array

        Inputs:
            x - training data (without first column of 1s)
            y - classification data
        Returns:
            array of weight adjustments for the next iteration
        """

        n_samples = x.shape[0]
        x_ext = np.concatenate((np.ones((n_samples, 1)), x), axis=1)

        return np.dot(y - self._activation(self._net_input(x)), x_ext)


class Perceptron(_LinearClassifier):
    """ Perceptron classifier

    Attributes:
        w - array of weights
    """

    def __init__(self):
        _LinearClassifier.__init__(self)

    def _cost(self, x, y):
        """ The cost function encapsulating the error in the model

        Inputs:
            x - training data (without first column of 1s)
            y - classification data
        Returns:
            value of the cost function, i.e., the number of errors in prediction
        """

        return (y - self._activation(self._net_input(x)) != 0).sum()

    def fit(self, x, y, n_iterations, learning_rate):
        """ Method to fit the model to given set of training data

        Inputs:
            x - array of training data (without first column of 1s)
            y - array of classifications for training data
            n_iterations - number of iterations of the algorithm to perform
            learning_rate - scale factor applied to updates of the weights array
        Returns:
            array of values of the cost function at each iteration
        """

        n_samples = x.shape[0]
        n_features = x.shape[1]
        cost_per_iteration = []

        self.w = np.random.normal(loc=0.0, scale=0.01, size=n_features + 1)

        for n in range(n_iterations):
            cost_per_iteration.append(self._cost(x, y))
            for i in range(n_samples):
                self.w += learning_rate * self._delta_w(x[i:i+1, :], y[i:i+1])

        return cost_per_iteration


class PerceptronAdaline(_LinearClassifier):
    """ Perceptron classifier

    Attributes:
        w - array of weights
        cost_per_iteration - array of the value of the cost function at each iteration
    """

    def __init__(self):
        _LinearClassifier.__init__(self)

    def _activation(self, x):
        """ Activation function
        Inputs:
            a - array
        Outputs:
            a - activation function for Adaline is identity function
        """

        return x

    def _cost(self, x, y):
        """ The cost function encapsulating the error in the model

        Inputs:
            x - training data (without first column of 1s)
            y - classification data
        Returns:
            value of the cost function J(w), i.e., sum of squared errors
        """

        return 0.5 * np.dot(y - self._net_input(x), y - self._net_input(x))

    def fit(self, x, y, n_iterations, learning_rate, mode):
        """ Method to fit the model to given set of training data

        Inputs:
            x - array of training data (without first column of 1s)
            y - array of classifications for training data
            n_iterations - number of iterations of the algorithm to perform
            learning_rate - scale factor applied to updates of the weights array
            mode - string, either "batch" or "stochastic"
        Returns:
            array of values of the cost function at each iteration
        """

        n_samples = x.shape[0]
        n_features = x.shape[1]
        cost_per_iteration = []

        self.w = np.random.normal(loc=0.0, scale=0.01, size=n_features + 1)

        if mode == "batch":
            for _ in range(n_iterations):
                cost_per_iteration.append(self._cost(x, y))
                self.w += learning_rate * self._delta_w(x, y)
        elif mode == "stochastic":
            for n in range(n_iterations):
                x, y = shuffle_training_examples(x, y)
                cost_per_iteration.append(self._cost(x, y))
                for i in range(n_samples):
                    self.w += learning_rate * self._delta_w(x[i:i+1, :], y[i:i+1])

        return cost_per_iteration
