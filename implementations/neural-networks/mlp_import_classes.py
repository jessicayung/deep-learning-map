"""
Multilayer Perceptron in Raw Python (with Numpy)
trained on Spiral Dataset

Not self-contained: imports classes and methods from `layers` and `data`.

Adapted from CS231n http://cs231n.github.io/neural-networks-case-study/

Jessica Yung
Jan 2018
"""
import numpy as np
import matplotlib.pyplot as plt
from layers import fc2d, relu, softmax_loss
from data import generate_spiral_data

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

X, y = generate_spiral_data(N, D, K, plot=False)

###########################
# Multilayer Perceptron classifier
###########################

# Initialise parameters
fc1 = fc2d(D, h1_units)
relu1 = relu()
fc2 = fc2d(h1_units, K)
data_loss_fn = softmax_loss(y)

for i in range(n_epochs):
    # Forward pass
    h1_prod = fc1.forward(X)
    h1 = relu1.forward(h1_prod)
    scores = fc2.forward(h1)
    data_loss = data_loss_fn.forward(scores)
    reg_loss = 0.5*reg*(np.sum(fc1.W * fc1.W) + np.sum(fc2.W*fc2.W))
    loss = data_loss + reg_loss
    if i % 1000 == 0:
        print("Epoch: %d, Loss: %f" % (i, loss))

    # Backprop
    # dLi/dfk = pk - 1(yi=k)
    dscores = data_loss_fn.backward()
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

exp_scores = np.exp(scores)
probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # sum along each row
print(probs.shape)
print(probs[:2])
# TODO: plot results

# TODO: split into training and validation sets

# TODO: plot training vs validation loss
