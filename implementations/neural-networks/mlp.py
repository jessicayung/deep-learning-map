import numpy as np
import matplotlib.pyplot as plt


def relu(X):
    return [[max(0, x) for x in example] for example in X]


def softmax(X):
    # TODO: optimise
    probs = []
    for example in X:
        exps = [np.exp(x) for x in example]
        den = sum(exps)
        probs.append(exps/den)
    return probs


def cross_entropy_loss(pred, true):
    if len(pred) != len(true):
        # TODO: rewrite exception
        raise Exception("Number of predictions different from number of labelled examples.")
    n_examples = len(pred)
    mean_loss = 0
    for i in range(n_examples):
        yh = pred[i]
        yt = true[i]
        loss = sum(-yt*np.log(yh))
        mean_loss += loss/n_examples
    return mean_loss


# Generate data
np.random.seed(42)
num_examples = 20
input_length = 3
int_range = 10
X = np.random.randint(int_range, size=[num_examples, input_length])
y = [x[0]+x[2]**2 for x in X]
y = [max(np.ceil(el/25), 4) - 1 for el in y]
print(X[:5])
print(y[:5])
# X, y = sklearn.datasets.make_moons(200, noise=0.20)
# plt.scatter(X[:,0], X[:,1], s=40, c=y)

# Set up an MLP with two hidden layers, 5 neurons in each layer
h1_units = 5
h2_units = 5
output_classes = 4

# Initialise weights
stdev = 0.001
W1 = np.random.randn(input_length, h1_units) * stdev
b1 = np.random.randn(h1_units) * stdev
# W2 = np.random.randn(h1_units, h2_units) * stdev
# b2 = np.random.randn(h2_units) * stdev
Wo = np.random.randn(h1_units, output_classes) * stdev
bo = np.random.randn(output_classes) * stdev

# Forward prop
h1_mul = np.dot(X, W1) + b1
print("h1 shape: ", h1.shape)
# print("h1: ", h1)
h1 = relu(h1_mul)
# h2 = np.dot(h1, W2) + b2
# print("h2: ", h2)
# h2 = relu(h2)
output = np.dot(h1, Wo) + bo
pred = softmax(output)
# print("output: ", output)
print(cross_entropy_loss(pred, y))


# Backward prop
dpred = sum([-yt * 1.0 / yh for yt, yh in zip(y.ravel(), pred.ravel())])
# sum_exps = sum([np.exp(sk) for sk in ...])
# doutput expr: (sum_exps - np.exp(sj))*np.exp(sj)/(sum_exps**2)
doutput = None # TODO: calculate, need to diff softmax
dWo = doutput * h1
dbo = doutput
dh1 = doutput * Wo
dh1_mul = dh1 * np.int64(h1 > 0)
db1 = dh1_mul
dW1 = X * dh1_mul
dX = W1 * dh1_mul