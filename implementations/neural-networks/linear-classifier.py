import numpy as np
import matplotlib.pyplot as plt

###########################
# Generate a spiral dataset (classes not linearly separable)
###########################

# Number of points per class
N = 100
# Number of dimensions of input
D = 2
# Number of classes
K = 3

# Initialise data matrix
X = np.zeros((N*K, D))
# Initialise vector of labels
y = np.zeros(N*K, dtype='uint8')
# Populate data matrix
for j in range(K):
    ix = range(N*j, N*(j+1))
    # Radius
    r = np.linspace(0.0, 1, N)  # np.linspace() returns evenly spaced numbers over an interval
    # Theta (diff interval for diff classes), with noise
    th = np.linspace(j*4, (j+1)*4, N) + np.random.randn(N)*0.2
    X[ix] = np.c_[r*np.sin(th), r*np.cos(th)]  # first arg = col1, second arg = col2
    y[ix] = j

plt.scatter(X[:, 0], X[:, 1], c=y, s=40)
plt.show()
