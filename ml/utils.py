
import numpy as np


def scale_features(x):
    """ Scale the features (columns) of a matrix
    Inputs:
        x - matrix
    Outputs:
        matrix with columns normalised

    For each column v, the normalised column v' is:
        v' = (v - v.mean()) / v.std(),
    where v.mean() is the mean and v.std() the standard deviation
    """

    n_features = x.shape[1]
    x_out = np.copy(x)  # put x into numpy

    # scale each column in turn
    for i in range(n_features):
        x_out[:, i] = (x_out[:, i] - x_out[:, i].mean()) / x_out[:, i].std()

    return x_out


def shuffle_training_examples(x, y):
    """ Shuffle the training examples (rows) of a training set
    Inputs:
        x - training data
        y - classifications of the training data
    Returns:
        x', y' - shuffled training data and matching classifications
    """
    n_samples = x.shape[0]
    assert len(y) == n_samples, "Length of x and y do not match"

    unified_dataset = np.concatenate((x, y.reshape(n_samples, 1)), axis=1)
    np.random.shuffle(unified_dataset)

    return unified_dataset[:, :-1], unified_dataset[:, -1]
