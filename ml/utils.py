
import numpy as np
import matplotlib.pyplot as plt

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

def plot_decision_regions(X, y, classifier, resolution=0.1):
    """ Plots the decision regions of a classifier with a two dimensional sample space
    Inputs:
        X - training or test data, with two columns
        y - classifications of the training/test data
        classifier - a method which returns the predicted class for any point in the 2-dim sample space
        resolution (optional) - resolution at which to plot the decision boundaries
    Returns:
        plot of the decision regions"""

    # check the input
    assert X.shape[1] == 2, "Samples should be two dimensional"

    # setup markers and colours
    markers = ('s', 'x', 'o')
    colours = ('red', 'blue', 'lightgreen')
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(colours[:len(np.unique(y))])

    # find minimum and maximum of the first two data columns, and construct a mesh grid to cover the sample space
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    # unravel the mesh grid and get the predicted class for each pair of values (x1, x2)
    x1_x2_pairs = np.array([xx1.ravel(), xx2.ravel()]).T
    Z = classifier.predict(x1_x2_pairs)

    # return Z to a grid shape and plot the contours
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)

    # add points to show the locations of the training/test examples
    for i, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=colours[i], marker=markers[i], label=cl, edgecolor='black')

    return plt
