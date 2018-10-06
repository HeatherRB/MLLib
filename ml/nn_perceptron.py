import numpy as np


class Perceptron:
    """ Perceptron classifier

    Input:
        activation_function - string, one of "step" or "Adaline"

    Attributes:
        w - array of weights
        cost_per_iteration - array of the value of the cost function at each iteration
    """

    def __init__(self, activation_function):

        # user selects a type of activation function
        activation_options = ["step", "Adaline"]
        assert activation_function in activation_options, "Not a valid activation function"
        self.activation_function = activation_function

        # initiate the array of weights
        self.w = None

    def _net_input(self, x):
        """ Function to calculate the net input z = x.w

        Input:
            x - array of training data (with first column of 1s)
        Returns:
            z = x.w
        """
        return np.dot(x, self.w)

    def predict(self, x):
        """ Function to predict the class of each sample (row) in x

        Inputs:
            x - training data (with first column of 1s)
        Returns:
            array of predicted classifications (values +1/-1)

        Predicted class of each sample v is:
            +1  if v.w >= 0
            -1  otherwise
        """
        return np.where(self._net_input(x) >= 0, 1, -1)

    def _delta_w(self, x, y):
        """ Function to calculate the (unscaled) change to the weights array

        Inputs:
            x - training data (with first column of 1s)
            y - classification data
        Returns:
            array of weight adjustments for the next iteration
        """
        if self.activation_function == "step":
            return np.dot(y - self.predict(x), x)
        elif self.activation_function == "Adaline":
            return np.dot(y - self._net_input(x), x)

    def _cost(self, x, y):
        """ The cost function encapsulating the error in the model

        Inputs:
            x - training data (with first column of 1s)
            y - classification data
        Returns:
            value of the cost function J(w)
        """

        if self.activation_function == "step":
            # cost is simply the number of errors in prediction
            return (y - self.predict(x) != 0).sum()
        elif self.activation_function == "Adaline":
            # sum of squared errors cost function
            return 0.5 * np.dot(y - self._net_input(x), y - self._net_input(x))

    def fit(self, training_data, y, n_iterations, learning_rate):
        """ Method to fit the model to given set of training data

        Inputs:
            training_data - array of training data
            y - array of classifications for training data
            n_iterations - number of iterations of the algorithm to perform
            learning_rate - scale factor applied to updates of the weights array
        Returns:
            array of values of the cost function at each iteration
        """

        n_samples = training_data.shape[0]
        n_features = training_data.shape[1]
        cost_per_iteration = []

        x = np.concatenate((np.ones((n_samples, 1)), training_data), axis=1)
        self.w = np.random.normal(loc=0.0, scale=0.01, size=n_features + 1)

        for _ in range(n_iterations):
            cost_per_iteration.append(self._cost(x, y))
            self.w += learning_rate * self._delta_w(x, y)

        return cost_per_iteration
