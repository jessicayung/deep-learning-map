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
        """
        if not (self.yw % 1) or not (self.yh % 1):
            print("yw: ", self.yw)
            print("yh: ", self.yh)
            raise Exception("stride not compatible")
        """

        self.yw, self.yh = int(self.yw), int(self.yh)

    def im2col(self):
        self.X_col = np.zeros((self.num_examples, self.fh*self.fw, self.yw*self.yh*self.Xd))
        print("X_col shape: ", self.X_col.shape)
        print("X shape: ", self.X.shape)
        self.W_row = np.zeros((self.num_filters,self.fh*self.fw*self.Xd))
        print("W_row shape: ", self.W_row.shape)
        print("W shape: ", self.W.shape)
        col = 0
        for i in range(self.yw):
            for j in range(self.yh):
                for k in range(self.num_examples):
                    if self.Xd == 1:
                        self.X_col[k,:,col] = np.reshape(self.X[k,j * self.sh:j * self.sh + self.fh, i * self.sw:i * self.sw + self.fw],-1)
                    else:
                        self.X_col[k,:,col] = np.reshape(self.X[k,j * self.sh:j * self.sh + self.fh, i * self.sw:i * self.sw + self.fw,:],-1)
                col += 1
        for n in range(self.num_filters):
            for i in range(self.fw):
                for j in range(self.yh):
                    self.W_row[n,:] = np.reshape(self.W[n,:,:,:],-1)
        return self.X_col, self.W_row

    def forward(self, X):
        self.X = X
        self.num_examples = len(self.X)
        self.y = np.zeros((self.yh, self.yw, self.num_filters))
        # print("Number of filters: ", num_filters)
        self.im2col()
        self.y = np.reshape(np.dot(self.W_row, self.X_col), (self.num_examples, self.yh, self.yw, self.num_filters))
        # print("Y shape: ", y.shape)
        # Add bias
        for n in range(self.num_filters):
            self.y[:,:,:,n] += self.b[n]
        return self.y

    def backward(self, dy):
        # TODO: check bias dims
        print("dy:",dy.shape)
        self.db = np.sum(dy, axis=(1,2)) # all except depth axis
        print("db:",self.db.shape)
        if self.Xd != 1:
            self.db = self.db.reshape(self.num_examples, self.num_filters, -1) # check if we need this.
            print("db:",self.db.shape)
        dy_reshaped = dy.reshape(self.num_examples, self.num_filters, -1)
        self.dW = np.array([np.dot(dy_reshaped[i], self.X_col[i].T) for i in range(self.num_examples)])
        print("dW:",self.dW.shape)
        self.dW = np.reshape(self.dW, (self.num_examples, self.num_filters, self.fh, self.fw, self.Xd))
        print("dW:",self.dW.shape)
        # reshape into num_filters, h, w, num_examples later
        W_reshape = self.W.reshape(self.num_filters, -1)
        dX_col = np.array([np.dot(W_reshape.T, dy_reshaped[i]) for i in range(self.num_examples)])
        print("dX_col:",dX_col.shape)
        return dX_col

