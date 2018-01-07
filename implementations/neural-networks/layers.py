"""
Neural Network Layers

Adapted from CS231n http://cs231n.github.io/neural-networks-case-study/

Jessica Yung
Jan 2018
"""
import numpy as np
import matplotlib.pyplot as plt

class fc2d:
    """Fully Connected Layer, 2D."""

    def __init__(self, example_length, units):
        self.units = units
        self.W = 0.01 * np.random.randn(example_length, units)
        self.b = np.zeros((1, units))

    def forward(self, X):
        """Forward pass.
        X: input to layer
        """
        self.X = X
        self.y = np.dot(X, self.W) + self.b
        return self.y

    def backward(self, dy):
        """Backward pass.
        NOTE: Assumes L2 regularisation.
        dy: gradient propagated to layer
        reg: L2 regularisation parameter
        """
        self.db = np.sum(dy, axis=0, keepdims=True)  # sum across columns
        self.dW = np.dot(self.X.T, dy)
        self.dX = np.dot(dy, self.W.T)
        return self.dX


class relu:
    def __init__(self):
        pass

    def forward(self, X):
        self.X = X
        return np.maximum(0,X)

    def backward(self, dy):
        dy[self.X <= 0] = 0
        return dy

class softmax_loss:
    def __init__(self, y):
        self.y = y

    def forward(self, X):
        self.num_examples = len(X)
        exp_scores = np.exp(X)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # sum along each row
        correct_logprobs = -np.log(probs[range(num_examples), self.y])
        data_loss = np.sum(correct_logprobs)/num_examples
        self.probs = probs
        return data_loss

    def backward(self, dy=None):
        # dLi/dfk = pk - 1(yi=k)
        if dy is None:
            dscores = self.probs
        else:
            dscores = dy
        dscores[range(num_examples),self.y] -= 1
        dscores /= self.num_examples
        return dscores

