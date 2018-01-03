"""
Adapted from Udacity's MiniFlow.
"""
import numpy as np


class Node:
    def __init__(self, inbound_nodes=[]):
        self.inbound_nodes = inbound_nodes
        self.value = None
        self.outbound_nodes = []
        self.gradients = {}
        for node in inbound_nodes:
            node.outbound_nodes.append(self)

    def forward(self):
        """
        Every node that uses this class as a base class will
        need to define its own 'forward' method.
        :return:
        """
        raise NotImplementedError

    def backward(self):
        """
        Every node that uses this class as a base class will
        need to define its own 'forward' method.
        :return:
        """
        raise NotImplementedError


class Input(Node):
    """An input to the network."""
    def __init__(self):
        Node.__init__(self)

    def forward(self):
        # Do nothing because nothing is calculated
        pass

    def backward(self):
        # TODO: (don't get this) Input node has no inputs, so gradient of node is zero
        # I get it if it's init at zero. but if it's got inputs, still init at zero?
        self.gradients = {self: 0}
        # Sum gradient from output nodes
        for n in self.outbound_nodes:
            self.gradients[self] += n.gradients[self]


class Linear(Node):
    """Node that performs a linear transform."""
    def __init__(self, X, W, b):
        Node.__init__(self, [X, W, b])

    def forward(self):
        X = self.inbound_nodes[0].value
        W = self.inbound_nodes[1].value
        b = self.inbound_nodes[2].value

        # output: each column is a unit, each row is an example
        self.value = np.dot(X, W) + b

    def backward(self):
        # init partial derivative for each inbound node
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}
        # sum gradients over all outputs
        for n in self.outbound_nodes:
            # df/dthis
            grad_cost = n.gradients[self]

            # X:
            # take coeff (W) and mult with grad cost in a way that preserves correct dims
            self.gradients[self.inbound_nodes[0]] = np.dot(grad_cost, self.inbound_nodes[1].value.T)
            # W
            self.gradients[self.inbound_nodes[0]] = np.dot(self.inbound_nodes[0].value.T, grad_cost)
            # b
            self.gradients[self.inbound_nodes[2]] += np.sum(grad_cost, axis=0, keepdims=False) # sum each column, i.e. sum across examples


class Sigmoid(Node):
    """Node that performs sigmoid activation function."""
    def __init__(self, node):
        Node.__init__(self, [node])

    def _sigmoid(self, x):
        """
        TODO: tbh can combine with forward method
        :param x: numpy array-like object
        :return: elementwise (TODO: check?) sigmoid of x
        """
        return 1. / (1. + np.exp(-x))

    def forward(self):
        input_value = self.inbound_nodes[0].value
        self.value = self._sigmoid(input_value)

    def backward(self):
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}
        # sum gradients over all outputs
        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]
            sigmoid = self.value
            self.gradients[self.inbound_nodes[0]] += grad_cost * sigmoid * (1-sigmoid)

class Layer:
    def __init__(self, units=0, W=None, activation="relu", n_inputs=0):
        if W is not None:
            self.units = len(W)
            self.n_inputs = len(W[0])
            self.W = W
        else:
            # TODO: make this more robust
            self.n_inputs = n_inputs
            self.units = units
            self.W = np.random.randn(self.units,self.n_inputs+1)
        self.W = W
        self.activation = activation

    def forward_pass(self,x):
        prod = np.matmul(self.W,x)
        return relu(prod)

    def backward_pass(self, grad):
        self.dW = grad.dot(x.T)
        self.dx = W.T.dot(grad)

def relu(array):
    return [max(0, x) for x in array]


W = [[1,2,3],[4,5,6],[7,8,9]]
x = [1,1,-1]

h1 = Layer(W)
print(h1.forward_pass(x))



