import numpy as np
import scipy.special as sps
import sys

class CostFunctions: # Class with different cost functions
    def __init__(self, function = "cross_entropy"):
        if function == 'cross_entropy' : # Suboptimal solution
            self.f   = self.cross_entropy
            self.df   = self.d_cross_entropy

        if function == 'mse' : # Suboptimal solution
            self.f    = self.mse
            self.df   = self.d_mse

    def __call__(self,tar, y, lmbd = 0, l2 = 0) :
        return self.f(tar, y, lmbd, l2), self.df(tar,y) # Returns chosen cost function

    def cross_entropy(self, tar, y,  lmbd = 0, l2 = 0): # Cross entropy loss function
        ce = -np.sum(sps.xlogy(y, tar) + sps.xlogy(1 - y, 1 - tar)) / tar.shape[0] + lmbd * l2
        return ce

    def d_cross_entropy(self, tar, y,  lmbd = 0):
        return tar - y

    def mse(self, tar, y,  lmbd = 0, l2 = 0) :         # Mean Squared error
        return 0.5*np.mean((tar-y)**2) + lmbd * l2

    def d_mse(self, tar, y,  lmbd = 0) :
        return tar-y
