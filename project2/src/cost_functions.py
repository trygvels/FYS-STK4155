import numpy as np

class CostFunctions: # Class with different cost functions
    def __init__(self, function = None):
        if function == 'cross_entropy' : # Suboptimal solution
            self.function    = self.cross_entropy

    def __call__(self, beta, X, y) :
        return self.function(beta, X, y) # Returns chosen cost function

    def cross_entropy(self, beta, X, y): # Cross entropy loss function
        b = np.dot(X,beta)
        reg = 0 # optional regularization term
        return -np.sum(y*b - np.log(1+np.exp(b))) + reg

    def mse(self, beta, X, y) :         # Mean Squared error
        b = np.dot(X,beta)
        return ((y - b)**2).mean() * 0.5

