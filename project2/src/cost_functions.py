import numpy as np
import sys
import scipy.special as sps

class CostFunctions: # Class with different cost functions
    def __init__(self, function = "cross_entropy"):
        if function == 'cross_entropy' : # Suboptimal solution
            self.f   = self.cross_entropy
            self.df   = self.d_cross_entropy
        if function == 'binary_cross_entropy' : # Suboptimal solution
            self.f   = self.bi_cross_entropy
            self.df   = self.d_bi_cross_entropy
        if function == 'mse' : # Suboptimal solution
            self.f    = self.mse
            self.df   = self.d_mse

    def __call__(self,tar, y, lmbd = 0, l2 = 0) :
        return self.f(tar, y, lmbd, l2), self.df(tar,y) # Returns chosen cost function

    def cross_entropy(self, tar, y,  lmbd = 0, l2 = 0): # Cross entropy loss function for part1
        ce = -np.sum(y*tar) + np.sum(np.log(1+np.exp(tar))) + lmbd*l2/len(tar)
        return ce

    def bi_cross_entropy(self, tar, y, lmbd = 0, l2 = 0):     
        y = y[:,1]
    
        ce = - (np.sum(y * np.log(1e-15+tar) + (1 - y)*np.log(1e-15+1 - tar)))/len(y) - (lmbd * l2)/len(tar)
        return ce

    def d_bi_cross_entropy(self, tar, y, lmbd = 0, l2 = 0):
        return (tar - y)/(tar*(1-tar))

    def d_cross_entropy(self, tar, y,  lmbd = 0):
        return tar - y

    def mse(self, tar, y,  lmbd = 0, l2 = 0) :         # Mean Squared error
        return 0.5*np.mean(( tar.ravel() - y.ravel() )**2) + lmbd * l2

    def d_mse(self, tar, y,  lmbd = 0) :
        return (tar-y)

    def R2(self, tar, y):
        ypred = tar.ravel()
        ytrue = y.ravel()
        R2 = 1. - np.sum((ytrue - ypred)**2)/np.sum((ytrue - np.mean(ytrue))**2)
        return R2
