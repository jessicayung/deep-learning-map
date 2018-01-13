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
        self.fh, self.fw = filter_shape
        self.num_filters = num_filters
        self.stride = stride
        self.padding = padding

        # Initialise bias
        self.b = np.zeros(self.num_filters)

        # Initialise weights
        self.W = 0.01 * np.random.randn(num_filters, self.fh, self.fw, self.Xd)
        self.sw = stride
        self.sh = stride
        self.yw = (self.Xw + 2 * padding - self.fw) / self.sw + 1
        self.yh = (self.Xh + 2 * padding - self.fh) / self.sh + 1
        self.yw, self.yh = int(self.yw), int(self.yh)

    def im2col(self):
        self.X_col = np.zeros((self.yw*self.yh*self.Xd,self.fh*self.fw))
        self.W_row = np.zeros((self.num_filters,self.fh*self.fw*self.Xd))
        col = 0
        for i in range(self.yw):
            for j in range(self.yh):
                self.X_col[:,col] = np.reshape(self.X[j * self.sh:j * self.sh + self.fh, i * self.sw:i * self.sw + self.fw,:],-1)
                col += 1
        for n in range(self.num_filters):
            for i in range(self.fw):
                for j in range(self.yh):
                    self.W_row[n,:] = np.reshape(self.W[n,:,:,:],-1)
        return self.X_col, self.W_row

    def forward(self, X):
        self.X = X
        self.y = np.zeros((self.yw, self.yh, self.num_filters))
        # print("Number of filters: ", num_filters)
        self.im2col()
        self.y = np.reshape(np.dot(self.W_row, self.X_col), (self.yh, self.yw, self.num_filters))
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
        self.dW = np.dot(dy_reshaped, self.X_col.T)
        # reshape into num_filters, h, w, num_examples later
        W_reshape = self.W.reshape(self.num_filters, -1)
        dX_col = np.dot(W_reshape, dy_reshaped)
        return dX_col

