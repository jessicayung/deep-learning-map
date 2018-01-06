"""
Convolutional Neural Network in Raw Python (with Numpy)
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
h1 = 100

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

# Initialise data matrix
X = np.zeros((num_examples, D))
# Initialise vector of labels
y = np.zeros(num_examples, dtype='uint8')
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
# plt.show()

# print("Initial loss should be about np.log(1.0/K) = ", np.log(1.0/K))

###########################
# CNN classifier
###########################


def conv(X, W, b, stride=1, padding=0):
    h, w, d = X.shape
    fh, fw = W[0].shape[:2]
    sw = stride
    sh = stride
    yw = (w + 2*padding - fw)/sw + 1
    yh = (h + 2*padding - fh)/sh + 1
    # print("Output shape: ", yw, yh)
    yw, yh = int(yw), int(yh)
    num_filters = len(W)
    y = np.zeros((yw, yh, num_filters))
    # print("Number of filters: ", num_filters)
    for n in range(num_filters):
        for i in range(yw):
            for j in range(yh):
                for depth in range(d):
                    y[j,i,n] += np.sum(W[n,:,:,depth] * X[j*sh:j*sh+fh,i*sw:i*sw+fw,depth])
                y[i,j,n] += b[n]
    # print("Y shape: ", y.shape)
    return y

X = np.array([[[1,0,0],[2,0,0],[0,0,0]],
                 [[0,0,0],[0,0,0],[0,0,0]],
                 [[0, 0, 0], [0, 0, 0], [3, 0, 0]]])
filter = np.array([[[1,0],[0,1]],[[1,0],[0,1]],[[1,0],[0,1]]])
W = np.array([[[[1,1,1],[0,0,0]],[[0,0,0],[1,1,1]]]])
print("W shape: ", W.shape)
print("X shape: ", X.shape)
b = np.array([0])
print(conv(X, W, b))

def im2col(X, W, stride=1, padding=0):
    # Convert each filter-sized region of X into a column vector
    h, w, d = X.shape
    fh, fw = W[0].shape[:2]
    sw = stride
    sh = stride
    yw = (w + 2 * padding - fw) / sw + 1
    yh = (h + 2 * padding - fh) / sh + 1
    # print("Output shape: ", yw, yh)
    yw, yh = int(yw), int(yh)
    num_filters = len(W)
    X_new = np.zeros((yw*yh*d,fh*fw))
    W_new = np.zeros((num_filters,fh*fw*d))
    print("X_new.shape: ", X_new.shape)
    print("W_new.shape: ", W_new.shape)
    # y = np.zeros((yw, yh, num_filters))
    col = 0
    for i in range(yw):
        for j in range(yh):
            X_new[:,col] = np.reshape(X[j * sh:j * sh + fh, i * sw:i * sw + fw,:],-1)
            col += 1
    for n in range(num_filters):
        for i in range(fw):
            for j in range(yh):
                W_new[n,:] = np.reshape(W[n,:,:,:],-1)
    return X_new, W_new


X_new, W_new = im2col(X, W)
print("X_new.shape: ", X_new.shape)
print("W_new.shape: ", W_new.shape)
print("X_new: ", X_new)
print("W_new: ", W_new)
"""
# Initialise parameters
W1 = 0.01 * np.random.randn(D, h1)
b1 = np.zeros((1, h1))
W2 = 0.01 * np.random.randn(h1, K)
b2 = np.zeros((1, K))

for i in range(n_epochs):
    # Forward pass
    h1_prod = np.dot(X, W1) + b1
    h1 = np.maximum(0, h1_prod)
    scores = np.dot(h1, W2) + b2
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # sum along each row
    correct_logprobs = -np.log(probs[range(num_examples), y])

    data_loss = np.sum(correct_logprobs)/num_examples
    reg_loss = 0.5*reg*(np.sum(W1 * W1) + np.sum(W2*W2))
    loss = data_loss + reg_loss
    if i % 1000 == 0:
        print("Epoch: %d, Loss: %f" % (i, loss))

    # Backprop
    # dLi/dfk = pk - 1(yi=k)
    dscores = probs
    dscores[range(num_examples),y] -= 1
    dscores /= num_examples

    db2 = np.sum(dscores, axis=0, keepdims=True)  # sum across columns
    dW2 = np.dot(h1.T, dscores)
    dW2 += reg * W2

    dh1 = np.dot(dscores, W2.T)
    dh1_prod = dh1
    # Backprop ReLU (gradient = 1 if > 0, = 0 otherwise)
    dh1_prod[h1 <= 0] = 0


    dW1 = np.dot(X.T, dh1_prod)
    dW1 += reg * W1
    db1 = np.sum(dh1_prod, axis=0, keepdims=True) # sum across columns

    # Parameter update
    W1 += -step_size * dW1
    b1 += -step_size * db1
    W2 += -step_size * dW2
    b2 += -step_size * db2

# Evaluate training set accuracy
h1 = np.maximum(0, np.dot(X, W1) + b1)
scores = np.dot(h1, W2) + b2
predicted_class = np.argmax(scores, axis=1)
print("Training accuracy: %.2f" % (np.mean(predicted_class == y)))

# TODO: plot results

# TODO: split into training and validation sets

# TODO: plot training vs validation loss
"""
