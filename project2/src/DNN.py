import sys
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, classification_report

from logreg      import LogReg
from initdata    import InitData
from activations import Activations

"""
This class is a feed-forward dense neural network used to train an arbitrary dataset.
"""


class NeuralNetwork:
    def __init__(
            self,
            X_data,
            Y_data,
            n_hidden_neurons=5,
            n_categories=2,
            epochs=1000,
            batch_size=100,
            eta=0.01,
            lmbd=0.0,
            activation="sigmoid"):

        self.X_data_full        = X_data
        self.Y_data_full        = Y_data

        self.n_inputs           = X_data.shape[0]
        self.n_features         = X_data.shape[1]

        self.n_hidden_neurons   = n_hidden_neurons
        self.n_categories       = n_categories

        self.act = Activations(activation)

        self.epochs             = epochs
        self.batch_size         = batch_size
        self.iterations         = self.n_inputs // self.batch_size
        self.eta                = eta
        self.lmbd               = lmbd

        self.create_biases_and_weights() # Initialize random weights and biases for two layers


    def create_biases_and_weights(self):
        # Calculate inital weights and biases for hidden layer
        self.hidden_weights = np.random.randn(self.n_features, self.n_hidden_neurons)
        self.hidden_bias    = np.zeros(self.n_hidden_neurons) + 0.01

        # Calculate initial weights and biases for output layer
        self.output_weights = np.random.randn(self.n_hidden_neurons, self.n_categories)
        self.output_bias    = np.zeros(self.n_categories) + 0.01

    def feed_forward(self): # Feed forward through full network
        ## feed-forward for training
        # calculate w*X + b
        self.z_h = np.matmul(self.X_data, self.hidden_weights) + self.hidden_bias
        # Pass through non-linear sigmoid gate
        self.a_h = self.act.f(self.z_h)

        # Calculate output output layer
        self.z_o = np.matmul(self.a_h, self.output_weights) + self.output_bias
        # Calculate probabolities from output layer
        self.probabilities = self.act.f(self.z_o)


    def feed_forward_out(self, X): # Run network without saving 
        # feed-forward for output
        z_h = np.matmul(X, self.hidden_weights) + self.hidden_bias
        a_h = self.act.f(z_h) #Activation function gate 

        # Output layer
        z_o = np.matmul(a_h, self.output_weights) + self.output_bias
        probabilities = self.act.softmax(z_o)

        return probabilities

    def backpropagation(self):
        error_output = self.probabilities - self.Y_data 
        error_hidden = np.matmul(error_output, self.output_weights.T) * self.act.df(self.a_h)

        self.output_weights_gradient    = np.matmul(self.a_h.T, error_output)
        self.output_bias_gradient       = np.sum(error_output, axis=0)

        self.hidden_weights_gradient    = np.matmul(self.X_data.T, error_hidden)
        self.hidden_bias_gradient       = np.sum(error_hidden, axis=0)

        if self.lmbd > 0.0: # Add regularization if lmbd value is given
            self.output_weights_gradient += self.lmbd * self.output_weights
            self.hidden_weights_gradient += self.lmbd * self.hidden_weights

        # Update weights 
        self.output_weights -= self.eta * self.output_weights_gradient
        self.output_bias    -= self.eta * self.output_bias_gradient
        self.hidden_weights -= self.eta * self.hidden_weights_gradient
        self.hidden_bias    -= self.eta * self.hidden_bias_gradient

    def predict(self, X):
        probabilities = self.feed_forward_out(X)
        return np.argmax(probabilities, axis=1)

    def predict_probabilities(self, X):
        probabilities = self.feed_forward_out(X)
        return probabilities

    def train(self):
        data_indices = np.arange(self.n_inputs)
   
        for i in range(self.epochs):
            for j in range(self.iterations):
                #if np.isnan(self.hidden_weights).any():
                #    print("found nan in",i,j)
                #    sys.exit()

                # pick datapoints with replacement
                chosen_datapoints = np.random.choice(
                    data_indices, size=self.batch_size, replace=False
                )

                # minibatch training data
                self.X_data = self.X_data_full[chosen_datapoints]
                self.Y_data = self.Y_data_full[chosen_datapoints]


                self.feed_forward()
                self.backpropagation()                