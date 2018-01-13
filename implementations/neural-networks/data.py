"""
Data generating and handling functions

Jessica Yung
Jan 2018
"""
import numpy as np
import matplotlib.pyplot as plt

def generate_spiral_data(points_per_class, input_dims=2, num_classes=3, plot=False):
    """Adapted from CS231n."""
    num_examples = points_per_class * num_classes
    # Initialise data matrix
    X = np.zeros((num_examples, input_dims))
    # Initialise vector of labels
    y = np.zeros(num_examples, dtype='uint8')
    # Populate data matrix
    for j in range(num_classes):
        ix = range(points_per_class * j, points_per_class * (j + 1))
        # Radius
        r = np.linspace(0.0, 1, points_per_class)  # np.linspace() returns evenly spaced numbers over an interval
        # Theta (diff interval for diff classes), with noise
        th = np.linspace(j * 4, (j+1) * 4, points_per_class) + np.random.randn(points_per_class) * 0.2
        X[ix] = np.c_[r*np.sin(th), r*np.cos(th)]  # first arg = col1, second arg = col2
        y[ix] = j
    if plot:
        if input_dims <= 2:
            plt.scatter(X[:, 0], X[:, 1], c=y, s=40)
            plt.show()
        else:
            print("Input dims > 2, cannot plot scatter diagram.")
    return X, y
