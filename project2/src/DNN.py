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
            Xtrain,
            Ytrain,
            Xtest = None,
            Ytest = None, 
            n_hidden_neurons=5,
            n_categories=2,
            epochs=1000,
            tol=0.001,
            batch_size=100,
            eta=0.01,
            lmbd=0.0,
            act_h="sigmoid", 
            act_o="softmax",
            cost="cross_entropy",
            nn_type = "classification",
            length = 1):

        print("---------------------------------------------------")
        print("Initializing", nn_type, "Neural Network")
        print("Cost function :", cost)
        print("---------------------------------------------------")
        self.nn_type = nn_type                      # Type of network

        self.Xtrain            = Xtrain
        self.Ytrain            = Ytrain
        self.Xtest             = Xtest
        self.Ytest             = Ytest

        self.n_inputs           = Xtrain.shape[0]   # Number of input data
        self.n_features         = Xtrain.shape[1]   # Number of features

        self.length = length                        # Number of free parameters
        self.n_hidden_neurons   = n_hidden_neurons  # Number of neurons in the hidden layer
        self.n_categories       = n_categories      # Number of outcome labels

        self.act_h_tag = act_h                      # Name of hidden layer activation function
        self.act_o_tag = act_o
        self.act_h = Activations(act_h)             # activation function of hidden layer
        self.act_o = Activations(act_o)             # activation function of output layer
        self.cost_tag = cost                        # Name of cost function
        self.cost = CostFunctions(cost)             # Cost function


        self.epochs             = epochs            # Number of epochs
        self.tol                = tol               # Convergence criteria
        self.batch_size         = batch_size        # Size of batch for SGD
        self.iterations         = self.n_inputs // self.batch_size # Number of iterations for SGD
        self.eta                = eta               # Learning rate
        self.lmbd               = lmbd              # Regularization parameter

        self.create_biases_and_weights() # Initialize random weights and biases for two layers

    def init_weight(self, n_in, n_out, tag):
        # Weight initialization function using optimized weights
        if tag == 'sigmoid':
            x = np.sqrt(6.0 / (n_in + n_out)) 
            return np.random.uniform(-x, x, size=(n_in, n_out))

        elif tag == 'tanh':
            x = 4.0 * np.sqrt(6.0 / (n_in + n_out)) 
            return np.random.uniform(-x, x, size=(n_in, n_out))

        elif tag == 'relu' or self.act_h_tag == 'elu':
            return np.random.randn(n_in, n_out)*np.sqrt(2.0/n_in)
        else:
            return np.random.randn(n_in, n_out)*np.sqrt(2.0/n_in)
            #return np.random.randn(n_in, n_out)

    def create_biases_and_weights(self):  
        # Calculate inital weights and biases for hidden layer
        self.hidden_weights = self.init_weight(self.n_features, self.n_hidden_neurons, self.act_h_tag)
        self.hidden_bias    = np.zeros(self.n_hidden_neurons)

        # Calculate initial weights and biases for output layer
        self.output_weights = self.init_weight(self.n_hidden_neurons, self.n_categories,self.act_h_tag)
        self.output_bias    = np.zeros(self.n_categories)

    def feed_forward(self): # Feed forward through full network
        ## feed-forward for training
        # calculate w*X + b
        self.z_h = np.matmul(self.Xtrain_batch, self.hidden_weights) + self.hidden_bias
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

    def nantest(self,x):
        if np.any(np.isnan(x)):
            print("Yep, its nan.")
            sys.exit()
        else:
            print("not nan")

    def backpropagation(self):
        # Calculate gradients for output layer
        if self.nn_type=="classification":
            error_output =  self.a_o - self.Ytrain_batch # For softmax we avoid divide by zero
        elif self.nn_type=="regression":               # In practice there is no difference here
            error_output =  self.cost.df(self.a_o, self.Ytrain_batch, self.lmbd) * self.act_o.df(self.z_o) 
        

        #self.nantest(error_output)
            
        self.output_weights_gradient    = np.matmul(self.a_h.T, error_output) 
        self.output_bias_gradient       = np.sum(error_output, axis=0)

        # Calculate gradients for hidden layer
        error_hidden = np.matmul(error_output, self.output_weights.T) * self.act_h.df(self.z_h) 
        self.hidden_weights_gradient    = np.matmul(self.Xtrain_batch.T, error_hidden)
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
        # returns 1d array of binary values
        a_o = self.feed_forward_out(X)
        return np.argmax(a_o, axis=1)

    def predict_a_o(self, X):
        # Returns both probabilities
        a_o = self.feed_forward_out(X)
        return a_o

    def score(self, ytrue, ypred):
        # Calculate score
        l2 = ( (self.hidden_weights**2).sum() + (self.output_weights**2).sum() ) / (2*self.n_features)
        cost = self.cost.f(ytrue, ypred, self.lmbd, l2)
        return cost
    
    def train(self):
        print("Training")

        data_indices = np.arange(self.n_inputs)
        self.costs = np.zeros((self.epochs,2))
        self.scores = np.zeros((self.epochs,2,2))
        
        nan = False
        for i in range(self.epochs):
            for j in range(self.iterations):
                
                # pick datapoints with replacement
                chosen_datapoints = np.random.choice( 
                    data_indices, size=self.batch_size, replace=False
                )

                # minibat ch training data
                self.Xtrain_batch = self.Xtrain[chosen_datapoints]
                self.Ytrain_batch = self.Ytrain[chosen_datapoints]
                
                # Run network once for this batch
                self.feed_forward()
                nan = self.backpropagation()          
                
            
                if nan: # Search in gradients, break if found
                    print("---------------------------------------------------")
                    print(f"NaN detected in gradients, epoch {i}.")
                    break
                
                                
                if self.nn_type=="classification":
                    self.costs[i,0] += self.score(self.predict(self.Xtrain),self.Ytrain)/(self.iterations*self.batch_size)
                    self.costs[i,1] += self.score(self.predict(self.Xtest),self.Ytest)/(self.iterations*self.batch_size)
                
                    # Save accuracy and roc_auc scores
                    self.scores[i,0,0] += accuracy_score(self.Ytrain[:,1], self.predict(self.Xtrain))/self.iterations
                    self.scores[i,0,1] += roc_auc_score( self.Ytrain[:,1], self.predict(self.Xtrain))/self.iterations
                    self.scores[i,1,0] += accuracy_score(self.Ytest[:,1], self.predict(self.Xtest))/self.iterations
                    self.scores[i,1,1] += roc_auc_score( self.Ytest[:,1], self.predict(self.Xtest))/self.iterations
            
                if self.nn_type=="regression":
                    self.costs[i,0] += self.score(self.predict_a_o(self.Xtrain),self.Ytrain)/(self.iterations*self.batch_size)
                    self.costs[i,1] += self.score(self.predict_a_o(self.Xtest),self.Ytest)/(self.iterations*self.batch_size)
                    # Save MSE and R2
                    self.scores[i,0,0] += self.costs[i,0]/self.iterations
                    self.scores[i,0,1] += self.cost.R2(self.predict_a_o(self.Xtrain), self.Ytrain)/self.iterations
                    self.scores[i,1,0] += self.costs[i,1]/self.iterations
                    self.scores[i,1,1] += self.cost.R2(self.predict_a_o(self.Xtest), self.Ytest)/self.iterations
        
            # Convergence test - Average change over 5 epochs
            if i > 10 or nan:
                tolerance =   np.abs( np.mean( self.costs[i-10:i-5,0] ) - np.mean( self.costs[i-5:i,0] ) )/ np.mean(self.costs[i-5:i,0]) 
                if tolerance < self.tol or nan:
                    print("---------------------------------------------------")
                    print("Convergence after {} epochs".format(i))
                    print("---------------------------------------------------")
                    self.costs = self.costs[:i+1,: ]
                    self.scores = self.scores[:i+1,:]
                    break

        return self.costs, self.scores

    def plot_costs(self,k=0):
        if k == 0:
            fig = plt.figure(figsize=(12,6))
        ax = plt.subplot(111)

        cmap = plt.get_cmap('tab20')
        c = cmap(np.linspace(0, 1, 20))[::-1] #self.length))

        if self.costs[-1,1] > 0.079: #0.7:
            a = 1
            ax.plot(self.costs[:,1], label=r"{:8s} LR: {:6}   $\lambda$: {:6}   Cost: {:.3f}".format(self.act_h_tag, "1e"+str(int(np.log10(self.eta))), "1e"+str(int(np.log10(self.lmbd))), self.costs[-1,1]), color=c[k], alpha = a, linewidth=1)
        else:
            a = 1
            ax.plot(self.costs[:,1], label=  r"{:8s} LR: {:6}   $\lambda$: {:8}".format(self.act_h_tag, "1e"+str(int(np.log10(self.eta))), "1e"+str(int(np.log10(self.lmbd))))\
                                                    + r"$\bf{Cost}$: " + r"{:.3f}    ".format( self.costs[-1,1]),
                                                    color=c[k], alpha = a, linewidth=3)

        #ax.loglog(self.costs[:,0], color=c[k], linestyle="--")

        
        ax.grid(True,linestyle=':')
        plt.gca().xaxis.grid(False)
        ax.grid(b=True, which='minor', linestyle=':', alpha=0.2)

        plt.xscale('symlog')
    
        plt.xlabel("Epoch")
        plt.ylabel("{}".format(self.cost_tag))
        #plt.xlim(1,100)
        #plt.ylim(24.5,27)
    
        chartBox = ax.get_position()
        if k == 0:
            ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.5, chartBox.height])
        else:
            ax.set_position([chartBox.x0, chartBox.y0, chartBox.width, chartBox.height])
        plt.legend(loc='upper center', bbox_to_anchor=(1.6, 1.0),prop={'family': 'monospace'})
        

    def plot_scores(self, k=0):
        if self.nn_type=="classification":
            # Color
            cmap = plt.get_cmap('tab20')
            c = cmap(np.linspace(0, 1, 20)) #self.length))
            #cmap = plt.get_cmap('tab10')
            #c = cmap(np.linspace(0, 1, 10)) #self.length))
            legend_properties = { 'family': 'monospace'}

            # Make fig
            if k == 0:
                fig = plt.figure(figsize=(12,6))
        
            # accuracy ----------
            ax1 = plt.subplot(211)
            plt.xscale('symlog')


            ax1.grid(True,linestyle=':')
            plt.gca().xaxis.grid(False)

            chartBox = ax1.get_position()
            if k == 0:
                ax1.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.5, chartBox.height])
            else:
                ax1.set_position([chartBox.x0, chartBox.y0, chartBox.width, chartBox.height])
            
            plt.ylabel("Accuracy")
            
            #ax1.set_ylim((0.80,0.825))
            #ax1.set_ylim((0.7,0.78))
            #ax1.set_xlim((1,13))

            # rocauc -----------
            ax2 = plt.subplot(212)
            chartBox = ax2.get_position()
            if k == 0:
                ax2.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.5, chartBox.height])
            else:
                ax2.set_position([chartBox.x0, chartBox.y0, chartBox.width, chartBox.height])
                           
            if self.scores[-1,1,1] < 0.735: # 0.82:
                a = 1
                #ax1.plot(self.scores[:,1,0], label=r"{:8s} LR: {:6}   $\lambda$: {:6}   Accuracy: {:.3f}    auc: {:.3f}".format(self.act_h_tag, "1e"+str(int(np.log10(self.eta))), "1e"+str(int(np.log10(self.lmbd))), self.scores[-1,1,0], self.scores[-1,1,1]), color=c[k], alpha = a, linewidth=1)
                ax1.plot(self.scores[:,1,0], label=r"{:8s} LR: {:6}   $N_h$: {:3d}   Accuracy: {:.3f}    auc: {:.3f}".format(self.act_h_tag, "1e"+str(int(np.log10(self.eta))), self.n_hidden_neurons, self.scores[-1,1,0], self.scores[-1,1,1]), color=c[k], alpha = a, linewidth=1)
                ax2.plot(self.scores[:,1,1], color=c[k], alpha = a, linewidth=1)
            else:
                a = 1
                #ax1.plot(self.scores[:,1,0], label=  r"{:8s} LR: {:6}   $\lambda$: {:8}".format(self.act_h_tag, "1e"+str(int(np.log10(self.eta))), "1e"+str(int(np.log10(self.lmbd))))\
                #                                        + r"$\bf{Accuracy}$: " + r"{:.3f}    ".format(self.scores[-1,1,0]) \
                #                                        + r"$\bf{auc} $: " + r"{:.3f}".format(self.scores[-1,1,1]),
                #                                       color=c[k], alpha = a, linewidth=3)
                ax1.plot(self.scores[:,1,0], label=  r"{:8s} LR: {:6} ".format(self.act_h_tag, "1e"+str(int(np.log10(self.eta))))\
                                                        + r"  $N_h$: " + r"{:3d}   ".format(self.n_hidden_neurons) \
                                                        + r"$\bf{Accuracy}$: " + r"{:.3f}    ".format(self.scores[-1,1,0]) \
                                                        + r"$\bf{auc} $: " + r"{:.3f}".format(self.scores[-1,1,1]),
                                                       color=c[k], alpha = a, linewidth=3)
                ax2.plot(self.scores[:,1,1], color=c[k], alpha = a, linewidth=3)



            ax1.legend(loc='upper center', bbox_to_anchor=(1.6, 1.0),prop=legend_properties)
            #ax2.plot(self.scores[:,0,1], label="Training ", color=c[k], linestyle="--")
            #ax2.set_ylim((0.7,0.74))#Class small
            #ax2.set_ylim((0.6,0.67))#Class big
            #xax2.set_xlim((1,13))
            plt.xscale('symlog')
            

            ax2.grid(True,linestyle=':')    
            plt.gca().xaxis.grid(False)

            plt.xlabel("Epoch")
            plt.ylabel("roc auc")
        else:

            # Color
            cmap = plt.get_cmap('tab20')
            c = cmap(np.linspace(0, 1, 20)) #self.length))
            legend_properties = { 'family': 'monospace'}

            # Make fig
            if k == 0:
                fig = plt.figure(figsize=(12,6))



            # MSE ----------
            ax1 = plt.subplot(211)
            ax1.grid(True,linestyle=':')
            ax1.grid(b=True, which='minor', linestyle=':', alpha=0.2)
            plt.gca().xaxis.grid(False, which='both')
            
            chartBox = ax1.get_position()
            if k == 0:
                ax1.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.5, chartBox.height])
            else:
                ax1.set_position([chartBox.x0, chartBox.y0, chartBox.width, chartBox.height])

            
            
            plt.ylabel("MSE")
            plt.xscale('symlog')
            ax1.set_ylim((0,0.02))


            # R2 -----------
            ax2 = plt.subplot(212)
            chartBox = ax2.get_position()
            if k == 0:
                ax2.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.5, chartBox.height])
            else:
                ax2.set_position([chartBox.x0, chartBox.y0, chartBox.width, chartBox.height])

            if self.scores[-1,1,1] <= 0.89:
                a = 1
                #ax1.plot(self.scores[:,1,0], label=r"{:8s} LR: {:6}   $\lambda$: {:6}   MSE: {:.3f}    R2: {:.3f}".format(self.act_h_tag, "1e"+str(int(np.log10(self.eta))), "1e"+str(int(np.log10(self.lmbd))), self.scores[-1,1,0], self.scores[-1,1,1]), color=c[k-1], alpha = a, linewidth=1)
                ax1.plot(self.scores[:,1,0], label=r"{:8s} LR: {:6}   $N_h$: {:3d}   MSE: {:.3f}    R2: {:.3f}".format(self.act_h_tag, "1e"+str(int(np.log10(self.eta))), self.n_hidden_neurons, self.scores[-1,1,0], self.scores[-1,1,1]), color=c[k-1], alpha = a, linewidth=1)
                ax2.plot(self.scores[:,1,1],  color=c[k-1], alpha = a, linewidth=1)
            else:
                a = 1
                #ax1.plot(self.scores[:,1,0], label=  r"{:8s} LR: {:6}   $\lambda$: {:8}".format(self.act_h_tag, "1e"+str(int(np.log10(self.eta))), "1e"+str(int(np.log10(self.lmbd))))\
                #                                        + r"$\bf{MSE}$: " + r"{:.3f}    ".format(self.scores[-1,1,0]) \
                #                                        + r"$\bf{R2} $:" + r"{:.3f}".format(self.scores[-1,1,1]),
                #                                       color=c[k-1], alpha = a, linewidth=3)
                ax1.plot(self.scores[:,1,0], label=  r"{:8s} LR: {:6} ".format(self.act_h_tag, "1e"+str(int(np.log10(self.eta))))\
                                                        + r"  $N_h$: " + r"{:3d}   ".format(self.n_hidden_neurons) \
                                                        + r"$\bf{MSE}$:" + r"{:.3f}   ".format(self.scores[-1,1,0]) \
                                                        + r" $\bf{R2}$: " + r"{:.3f}".format(self.scores[-1,1,1]),
                                                       color=c[k-1], alpha = a, linewidth=3)
                ax2.plot(self.scores[:,1,1],color=c[k-1], alpha = a, linewidth=3)

            ax1.legend(loc='upper center', bbox_to_anchor=(1.6, 1.0),prop=legend_properties)
            #ax2.plot(self.scores[:,0,1], label="Training ", color=c[k], linestyle="--")
            ax2.set_ylim((0.2,1))
            plt.xscale('symlog')

            ax2.grid(True,linestyle=':')    
            ax2.grid(b=True, which='minor', linestyle=':', alpha=0.4)
            plt.gca().xaxis.grid(False, which='both')
            

            plt.xlabel("Epoch")
            plt.ylabel("R2")