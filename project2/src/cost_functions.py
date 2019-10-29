import numpy as np

class CostFunctions:
    def __init__(self, function = None):
        self.function = function if (function is not None) else 'mse'
        if self.function == 'cross_entropy' :
            self.function    = self.cross_entropy

    def __call__(self, beta, X, y) :
        return self.function(beta, X, y) #Returns chosen cost function

    def cross_entropy(self, beta, X, y):
        # Cross-entropy loss for logistic regression:
        b = np.dot(X,beta)
        reg = 0 # Regularization term
        return -np.sum(y*b - np.log(1+np.exp(b))) + reg

    def mse(self, beta, X, y) :
        b = np.dot(X,beta)
        return ((y - b)**2).mean() * 0.5

