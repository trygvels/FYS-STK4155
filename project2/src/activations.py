import numpy as np

class Activations():
    def __init__(self, alpha = 1):
        self.alpha = alpha

    # Sigmoid
    def sigmoid(self, z):
        return 1.0/(1.0 + np.exp(-z))

    def dsigmoid(self, z):
        return self.sigmoid(z)*(1.0 - self.sigmoid(z))

    # Softmax
    def softmax(self, z):
        exp_term = np.exp(z)
        return exp_term / np.sum(exp_term, axis=1, keepdims=True)
    
    # tanh
    def tanh(self, z):
        return np.tanh(z)

    def dtanh(self, z):
        return 1 - self.tanh(z)**2

    # Relu
    def relu(self, z):
        return self.alpha*np.maximum(z, 0)
    
    def drelu(self, z):
        return self.alpha * (z > 0)

    # elu
    def elu(self, z):
        return np.choose(z < 0, [z, self.salpha*(np.exp(z)-1)])
        
    def delu(self, z):
        return np.choose(x > 0, [1, self.alpha * np.exp(z)])




