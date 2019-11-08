import sys
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, classification_report
from sklearn.preprocessing import OneHotEncoder

from logreg         import LogReg
from initdata       import InitData
from activations    import Activations
from cost_functions import CostFunctions

"""
This class is a feed-forward dense neural network used to train an arbitrary dataset.
"""


class NeuralNetwork:
    def __init__(
            self,
            Xdata,
            Ydata,
            n_hidden_neurons=5,
            n_categories=2,
            epochs=1000,
            batch_size=100,
            eta=0.01,
            lmbd=0.0,
            act_h="sigmoid", 
            act_o="softmax",
            cost="cross_entropy",
            nn_type = "classification"):

        print("---------------------------------------------------")
        print("Initializing", nn_type, "Neural Network")
        print("Cost function :", cost)
        print("---------------------------------------------------")
        self.nn_type = nn_type

        self.Xdata_full        = Xdata
        self.Ydata_full        = Ydata

        self.n_inputs           = Xdata.shape[0]
        self.n_features         = Xdata.shape[1]

        self.n_hidden_neurons   = n_hidden_neurons
        self.n_categories       = n_categories

        self.act_h_tag = act_h
        self.act_h = Activations(act_h)
        self.act_o = Activations(act_o)
        self.cost = CostFunctions(cost)


        self.epochs             = epochs
        self.batch_size         = batch_size
        self.iterations         = self.n_inputs // self.batch_size
        self.eta                = eta
        self.lmbd               = lmbd

        self.create_biases_and_weights() # Initialize random weights and biases for two layers

    def init_weight(self,n_in,n_out):
        # Weight initialization function
        if self.act_h_tag == 'sigmoid':
            x = np.sqrt(6.0 / (n_in + n_out))
            return np.random.uniform(-x, x, size=(n_in, n_out))

        elif self.act_h_tag == 'tanh':
            x = 4.0 * np.sqrt(6.0 / (n_in + n_out))
            return np.random.uniform(-x, x, size=(n_in, n_out))

        elif self.act_h_tag == 'relu' or self.act_h_tag == 'elu':
            return np.random.randn(n_in, n_out)*np.sqrt(2.0/n_in)

        else:
            return np.random.randn(n_in, n_out)

    def create_biases_and_weights(self):  
        # Calculate inital weights and biases for hidden layer
        self.hidden_weights = self.init_weight(self.n_features, self.n_hidden_neurons) # np.random.randn(self.n_features, self.n_hidden_neurons)* np.sqrt(2.0 / self.n_features)
        self.hidden_bias    = np.zeros(self.n_hidden_neurons)

        # Calculate initial weights and biases for output layer
        self.output_weights = self.init_weight(self.n_hidden_neurons, self.n_categories) #  np.random.randn(self.n_hidden_neurons, self.n_categories)* np.sqrt(2.0 / self.n_features)
        self.output_bias    = np.zeros(self.n_categories)

    def feed_forward(self): # Feed forward through full network
        ## feed-forward for training
        # calculate w*X + b
        self.z_h = np.matmul(self.Xdata, self.hidden_weights) + self.hidden_bias
        # Pass through non-linear sigmoid gate
        self.a_h = self.act_h.f(self.z_h)

        # Calculate output output layer
        self.z_o = np.matmul(self.a_h, self.output_weights) + self.output_bias
        # Calculate probabolities from output layer
        self.a_o = self.act_o.f(self.z_o)


    def feed_forward_out(self, X): # Run network without saving 
        # feed-forward for output
        z_h = np.matmul(X, self.hidden_weights) + self.hidden_bias
        a_h = self.act_h.f(z_h) #Activation function gate 

        # Output layer - If reg different cost
        z_o = np.matmul(a_h, self.output_weights) + self.output_bias
        a_o = self.act_o.f(z_o)

        return a_o

    def backpropagation(self):
        """
        TODO:
        Implement multilayer
        """

        # Calculate gradients for output layer
        error_output = (self.a_o - self.Ydata)
        #error_output =  self.cost.df(self.a_o,self.Ydata, self.lmbd) * self.act_o.df(self.z_o)

        self.output_weights_gradient    = np.matmul(self.a_h.T, error_output) 
        self.output_bias_gradient       = np.sum(error_output, axis=0)

        # Calculate gradients for hidden layer        
        error_hidden = np.matmul(error_output, self.output_weights.T) * self.act_h.df(self.z_h) 
        self.hidden_weights_gradient    = np.matmul(self.Xdata.T, error_hidden)
        self.hidden_bias_gradient       = np.sum(error_hidden, axis=0)
    

        if self.lmbd > 0.0: # Add regularization if lmbd value is given
            self.output_weights_gradient += self.lmbd * self.output_weights
            self.hidden_weights_gradient += self.lmbd * self.hidden_weights

        # Checking for NaNs
        if np.any(np.isnan(self.hidden_weights_gradient)) or np.any(np.isnan(self.output_weights_gradient)) \
            or np.any(np.isnan(self.hidden_bias_gradient)) or np.any(np.isnan(self.output_bias_gradient)):
            return True
            
        # Update weights 
        self.output_weights -= self.eta * self.output_weights_gradient
        self.output_bias    -= self.eta * self.output_bias_gradient
        self.hidden_weights -= self.eta * self.hidden_weights_gradient
        self.hidden_bias    -= self.eta * self.hidden_bias_gradient

    def predict(self, X):
        # returns 1d array
        a_o = self.feed_forward_out(X)
        return np.argmax(a_o, axis=1)

    def predict_a_o(self, X):
        # Returns both probabilities
        a_o = self.feed_forward_out(X)
        return a_o

    def score(self, ytrue, ypred):
        l2 = ( (self.hidden_weights**2).sum() + (self.output_weights**2).sum() ) / (2*self.n_features)
        cost = self.cost.f(ytrue, ypred, self.lmbd, l2)
        return cost

    def train(self):
        data_indices = np.arange(self.n_inputs)
        self.costs = np.zeros(self.epochs)
        nan = False
        for i in range(self.epochs):
            for j in range(self.iterations):
                
                # pick datapoints with replacement
                chosen_datapoints = np.random.choice( 
                    data_indices, size=self.batch_size, replace=False
                )

                # minibat ch training data
                self.Xdata = self.Xdata_full[chosen_datapoints]
                self.Ydata = self.Ydata_full[chosen_datapoints]

                self.feed_forward()
                nan = self.backpropagation()          
                if nan: # Search in gradients, break if found
                    print("---------------------------------------------------")
                    print(f"NaN detected in gradients, epoch {i}.")
                    break

                self.costs[i] += self.score(self.predict_a_o(self.Xdata),self.Ydata)
           
            if nan:
                break

            # Convergence test - Average change over 5 epochs
            if i > 10:
                tolerance =   np.abs( np.mean( self.costs[i-10:i-5] ) - np.mean( self.costs[i-5:i] ) )/ np.mean(self.costs[i-5:i]) 
                if tolerance < 0.001:
                    print("---------------------------------------------------")
                    print("Convergence after {} epochs".format(i))
                    print("---------------------------------------------------")
                    self.costs[i:] = None
                    break

        return self.costs

    def plot_costs(self):
        #plt.semilogy(self.costs)
        plt.plot(self.costs)
        plt.show()