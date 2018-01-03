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



