import numpy as np

class CostFunctions: # Class with different cost functions
    def __init__(self, function = "cross_entropy"):
        if function == 'cross_entropy' : # Suboptimal solution
            self.function    = self.cross_entropy

        if function == 'mse' : # Suboptimal solution
            self.function    = self.mse

    def __call__(self,tar, y) :
        return self.function(tar, y) # Returns chosen cost function

    # DERIVATIVES FOR BOTH CROSS ENTROPY AND MSE IS tar-y

    def cross_entropy(self, tar, y): # Cross entropy loss function
        reg = 0 # optional regularization term
        return -np.sum(y*tar - np.log(1+np.exp(tar))) + reg

    def mse(self, tar, y) :         # Mean Squared error
        return ((y - tar)**2).mean() * 0.5