"""
Convolutional Neural Network Layer in Raw Python (with Numpy, implemented using im2col)

WARNING: Not yet tested

Jessica Yung
Jan 2018
"""
import numpy as np
import matplotlib.pyplot as plt

class cnn2d:
    def __init__(self, input_shape, filter_shape, num_filters, stride=1, padding=0):
        self.Xh, self.Xw, self.Xd = input_shape
        self.fh, self,fw = filter_shape
        self.num_fiters = num_filters
        self.stride = stride
        self.padding = padding

        # Initialise bias
        self.b = np.zeros(self.num_filters)

        # Initialise weights
        self.W = 0.01 * np.random.randn(num_filters, fh, fw, self.Xd)
        self.sw = stride
        self.sh = stride
        self.yw = (w + 2 * padding - fw) / sw + 1
        self.yh = (h + 2 * padding - fh) / sh + 1
        self.yw, self.yh = int(self.yw), int(self.yh)

    def im2col(self,X):
        self.X_new = np.zeros((self.yw*self.yh*self.Xd,self.fh*self.fw))
        self.W_new = np.zeros((self.num_filters,self.fh*self.fw*self.Xd))
        col = 0
        for i in range(self.yw):
            for j in range(self.yh):
                self.X_new[:,col] = np.reshape(self.X[j * self.sh:j * self.sh + self.fh, i * self.sw:i * self.sw + self.fw,:],-1)
                col += 1
        for n in range(self.num_filters):
            for i in range(self.fw):
                for j in range(self.yh):
                    self.W_new[n,:] = np.reshape(self.W[n,:,:,:],-1)
        return self.X_new, self.W_new

    def forward(self, X):
        self.X = X
        self.y = np.zeros((self.yw, self.yh, self.num_filters))
        # print("Number of filters: ", num_filters)
        self.im2col(self.X, self.W)
        self.y = np.reshape(np.dot(self.W_new, self.X_new), (self.yh, self.yw, self.num_filters))
        # print("Y shape: ", y.shape)
        # Add bias
        for n in range(self.num_filters):
            self.y[:,:,n] += self.b[n]
        return self.y

    def backward(self, dy):
        # TODO: check bias dims
        self.db = np.sum(dy, axis=(0,1)) # all except depth axis
        self.db = self.db.reshape(self.num_filters, -1) # check if we need this.
        dy_reshaped = self.dy.reshape(self.num_filters, -1)
        self.dW = np.dot(dy_reshaped, X_new.T)
        # reshape into num_filters, h, w, num_examples later
        W_reshape = self.W.reshape(self.num_filters, -1)
        dX_col = np.dot(W_reshape, dy_reshaped)
        return dX_col

