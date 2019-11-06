import numpy as np
import random

class Activations():
    def __init__(self, activation="sigmoid"):
        self.getactivation(activation)

    def getactivation(self, name) :
        if name == 'sigmoid' :
            self.f = self.sigmoid
            self.df = self.dsigmoid
        elif name == 'tanh' :
            self.f = self.tanh
            self.df = self.dtanh
        elif name == 'relu' :
            self.f = self.relu
            self.df= self.drelu
        elif name == 'elu' :
            self.f = self.elu
            self.df = self.delu
        elif name == 'softmax' :
            self.f = self.softmax
            self.df = self.dsoftmax
        elif name == 'identity' :
            self.f = self.identity
            self.df = self.didentity
        else :
            raise ValueError("Did not find activation: " + str(name))

    # Sigmoid
    def sigmoid(self, z):
        return 1.0/(1.0 + np.exp(-z))

    def dsigmoid(self, z):
        return self.sigmoid(z)*(1 - self.sigmoid(z))

    # tanh
    def tanh(self, z):
        return np.tanh(z)

    def dtanh(self, z):
        return 1 - self.tanh(z)**2

    # Relu
    def relu(self, z):
        return np.maximum(z, 0)
    
    def drelu(self, z):
        return (z > 0)

    # elu
    def elu(self, z):
        return np.choose(z < 0, [z, (np.exp(z)-1)])
        
    def delu(self, z):
        return np.choose(z > 0, [1,  np.exp(z)])

    # Softmax
    def softmax(self, z):
        exp_term = np.exp(z)
        return exp_term / np.sum(exp_term, axis=1, keepdims=True)

    def dsoftmax(self, z):
        return self.softmax(z) * (1 - self.softmax(z)) 

    # identity
    def identity(self, z):
        return z

    def didentity(self, z):
        return 1

