import numpy as np

class CostFunctions: # Class with different cost functions
    def __init__(self, function = "cross_entropy"):
        if function == 'cross_entropy' : # Suboptimal solution
            self.f   = self.cross_entropy
            self.df   = self.d_cross_entropy

        if function == 'mse' : # Suboptimal solution
            self.f    = self.mse
            self.df   = self.d_mse

    def __call__(self,tar, y) :
        return self.f(tar, y), self.df(tar,y) # Returns chosen cost function


    # DERIVATIVES FOR BOTH CROSS ENTROPY AND MSE IS tar-y

    def cross_entropy(self, tar, y): # Cross entropy loss function
        reg = 0 # optional regularization term
        return -np.sum(y*tar - np.log(1+np.exp(tar))) + reg

    def d_cross_entropy(self, tar, y) :
        #return -np.sum(target/y - (1-target)/(1-y))
        return tar - y #y - tar

    def mse(self, tar, y) :         # Mean Squared error
        return 0.5*np.mean((tar-y)**2) #+ (lmbd * w**2))

    def d_mse(self, tar, y) :
        return tar-y
