"""
Multilayer Perceptron in Raw Python (with Numpy)
trained on Spiral Dataset

Adapted from CS231n http://cs231n.github.io/neural-networks-case-study/

Jessica Yung
Jan 2018
"""
import numpy as np
import matplotlib.pyplot as plt

###########################
# TODO: set model parameters
###########################

# Parameters
# regularisation strength
reg = 1e-3
# Parameter update step size
step_size = 1e-0
# Number of training epochs
n_epochs = 9000
# Size of hidden layer
h1_units = 100

###########################
# Generate a spiral dataset (classes not linearly separable)
###########################

# Number of points per class
N = 100
# Number of dimensions of input
D = 2
# Number of classes
K = 3
num_examples = N*K

def generate_spiral_data(points_per_class, input_dims=2, num_classes=3, plot=False):
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

class fc2d:

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

X, y = generate_spiral_data(N, 2, 3, plot=False)

# print("Initial loss should be about np.log(1.0/K) = ", np.log(1.0/K))

###########################
# Multilayer Perceptron classifier
###########################

# Initialise parameters
fc1 = fc2d(D, h1_units)
relu1 = relu()
fc2 = fc2d(h1_units, K)

for i in range(n_epochs):
    # Forward pass
    h1_prod = fc1.forward(X)
    h1 = relu1.forward(h1_prod)
    scores = fc2.forward(h1)
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # sum along each row
    correct_logprobs = -np.log(probs[range(num_examples), y])

    data_loss = np.sum(correct_logprobs)/num_examples
    reg_loss = 0.5*reg*(np.sum(fc1.W * fc1.W) + np.sum(fc2.W*fc2.W))
    loss = data_loss + reg_loss
    if i % 1000 == 0:
        print("Epoch: %d, Loss: %f" % (i, loss))

    # Backprop
    # dLi/dfk = pk - 1(yi=k)
    dscores = probs
    dscores[range(num_examples),y] -= 1
    dscores /= num_examples
    dh1 = fc2.backward(dscores)
    dh1_prod = relu1.backward(dh1)
    dX = fc1.backward(dh1_prod)

    # Gradient from regularisation
    fc1.dW += reg*fc1.W
    fc2.dW += reg*fc2.W

    # Parameter update
    fc1.W += -step_size * fc1.dW
    fc1.b += -step_size * fc1.db
    fc2.W += -step_size * fc2.dW
    fc2.b += -step_size * fc2.db

# Evaluate training set accuracy
h1_prod = fc1.forward(X)
h1 = relu1.forward(h1_prod)
scores = fc2.forward(h1)
predicted_class = np.argmax(scores, axis=1)
print("Training accuracy: %.2f" % (np.mean(predicted_class == y)))

# TODO: plot results

# TODO: split into training and validation sets

# TODO: plot training vs validation loss
