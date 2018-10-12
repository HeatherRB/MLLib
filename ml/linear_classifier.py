
import numpy as np
from ml.utils import *


class _LinearClassifier:
    """Base class for a linear classifier

    Attributes:
        w - array of weights
        random_state - seed for random starting weights
    """

    def __init__(self, random_state):

        # initiate the array of weights
        self.w = None
        self.random_state = random_state

    def _net_input(self, x):
        """ Net input

        Input:
            x - array of training data (without first column of 1s)
        Returns:
            net input z = x.w
        """

        n_samples = x.shape[0]
        x_ext = np.concatenate((np.ones((n_samples, 1)), x), axis=1)

        return x_ext.dot(self.w)

    def _activation(self, z):
        """ Default activation function (perceptron)
        Inputs:
            z - scalar/array
        Outputs:
            value of perceptron activation function
            phi(z) =    +1  if z >= 0
                        -1  otherwise
        """

        return np.where(z >= 0, 1, -1)

    def _cost(self, x, y):
        """ Default cost function (perceptron) encapsulating the error in the model

        Inputs:
            x - training data (without first column of 1s)
            y - classification data
        Returns:
            value of the cost function, i.e., the number of errors in prediction
        """

        return (y - self._activation(self._net_input(x)) != 0).sum()

    def predict(self, x, a=0):
        """ Function to predict the class of each sample (row) in x

        Inputs:
            x - training data (without first column of 1s)
            a (optional) - threshold value above which prediction is positive
        Returns:
            array of predicted classifications (values +1/-1)

        Predicted class of each sample is:
            +1  if phi(z) >= a
            -1  otherwise,
            where z = x.w
        """

        return np.where(self._activation(self._net_input(x)) >= a, 1, -1)

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

        return (y - self._activation(self._net_input(x))).dot(x_ext)

    def fit_stochastic(self, x, y, n_iterations, learning_rate):
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

        rgen = np.random.RandomState(self.random_state)
        self.w = rgen.normal(loc=0.0, scale=0.01, size=n_features + 1)

        for n in range(n_iterations):
            cost_per_iteration.append(self._cost(x, y))
            x, y = shuffle_training_examples(x, y)
            for i in range(n_samples):
                self.w += learning_rate * self._delta_w(x[i:i+1, :], y[i:i+1])

        return cost_per_iteration

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

        n_features = x.shape[1]
        cost_per_iteration = []

        self.w = np.random.normal(loc=0.0, scale=0.01, size=n_features + 1)

        for _ in range(n_iterations):
            cost_per_iteration.append(self._cost(x, y))
            self.w += learning_rate * self._delta_w(x, y)

        return cost_per_iteration


class Perceptron(_LinearClassifier):
    """ Perceptron classifier

    This classifier simply adopts the default activation and cost functions from the base class

    Attributes:
        w - array of weights
        random_state - seed for random starting weights
    """

    def __init__(self, random_state):
        _LinearClassifier.__init__(self, random_state)


class PerceptronAdaline(_LinearClassifier):
    """ Perceptron classifier

    Attributes:
        w - array of weights
        random_state - seed for random starting weights
    """

    def __init__(self, random_state):
        _LinearClassifier.__init__(self, random_state)

    def _activation(self, z):
        """ Activation function
        Inputs:
            z - array
        Outputs:
            activation function for Adaline perceptron
            phi(z) = z
        """

        return z

    def _cost(self, x, y):
        """ The cost function encapsulating the error in the model

        Inputs:
            x - training data (without first column of 1s)
            y - classification data
        Returns:
            value of the cost function, i.e., sum of squared errors
            J(w) = 1/2 (y - phi(z)).(y - phi(z))
        """

        return 0.5 * (y - self._net_input(x)).dot(y - self._net_input(x))


class LogisticRegression(_LinearClassifier):
    """ Logistic regression classifier

    Attributes:
        w - array of weights
        random_state - seed for random starting weights
        c - regularisation parameter
    """

    def __init__(self, random_state, c):
        _LinearClassifier.__init__(self, random_state)
        self.c = c

    def _activation(self, z):
        """ Activation function for logistic regression

        Inputs:
            z - array
        Returns:
            phi(x) = 1/(1 + e^{-z})
            """

        return 1 / (1 + np.exp(-z))

    def _cost(self, x, y):
        """ Cost function for logistic regression

        Inputs:
            x - training data (without first column of 1s)
            y - classification data
        Returns:
            value of the regularised cost function, i.e., (negative) log-likelihood function
            J(w) = - y.log(phi(z)) - (1 - y).log(1 - phi(z)) + w.w
            """

        phi_z = self._activation(self._net_input(x))

        return -y.dot(np.log(phi_z)) - (1 - y).dot(np.log(1 - phi_z)) + 0.5 * self.c * self.w.dot(self.w)

    def predict(self, x, a=0.5):
        """ Function to predict the class of each sample (row) in x

        Inputs:
            x - training data (without first column of 1s)
            a (optional) - threshold value above which prediction is positive
        Returns:
            array of predicted classifications (values +1/-1)

        Predicted class of each sample is:
            +1  if phi(z) >= a
            0   otherwise,
            where z = x.w
        """

        return np.where(self._activation(self._net_input(x)) >= a, 1, 0)

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

        return (y - self._activation(self._net_input(x))).dot(x_ext) - self.c * self.w