import sys
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, classification_report
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import GridSearchCV
plt.style.use(u"~/.matplotlib/stylelib/trygveplot_astro.mplstyle")

from logreg     import LogReg
from initdata   import InitData
from DNN        import NeuralNetwork
"""
In this part of the project, we assess the predictive ability of a feed-forward Neural 
network on determining default based on credit card data. 
"""
## Get data from InitData Class
data = InitData()
XTrain, yTrain, XTest, yTest, Y_train_onehot, Y_test_onehot =  data.credit_data(trainingShare=0.5, return_cols=False)#, drop_zero=True,drop_neg2=True)
logreg = LogReg() # init Logreg class

## Params
explore = False
sklearn = False
metric = "accuracy" #"roc_auc"
cost = "cross_entropy"

if explore==True: # Explore parameter space for credit card data
    # Try it all
    eta_vals = np.logspace(-5, 1, 7)
    lmbd_vals = np.logspace(-5, 1, 7)
    acts_hidden = ["sigmoid", "tanh", "relu", "elu"]
    hidden_neurons = [4,8,12,16,50,100] 
    epochs = 20

    # Faster
    eta_vals = np.logspace(-3, 0, 4)
    lmbd_vals = np.logspace(-3, 0, 4)
    acts_hidden = ["sigmoid", "relu"]
    hidden_neurons = [4,8,12,16,50] 
    epochs = 20

else: # Optimal setup for credit card using all data
    # Best relu
    eta_vals = [0.0001]
    lmbd_vals = [0.1]
    acts_hidden = ["relu"]
    hidden_neurons = [16] 
    epochs=200
    
    # GOAT
    lmbd_vals = [0.1]
    eta_vals = [0.01]
    acts_hidden = ["sigmoid"]
    hidden_neurons = [12] 
    epochs=200
    

if sklearn == False:
    # grid search
    best_accuracy = 0
    best_rocauc = 0
    for i, eta in enumerate(eta_vals):
        for j, lmbd in enumerate(lmbd_vals):
            for k, act_h in enumerate(acts_hidden):
                for l, hn in enumerate(hidden_neurons):
                    dnn = NeuralNetwork(XTrain, Y_train_onehot.toarray(), cost =cost,  eta=eta, lmbd=lmbd, n_hidden_neurons=hn, act_h=act_h, epochs=epochs)
                    costs = dnn.train()
                    
                    # Probabilities for cost calculation
                    print(dnn.score(dnn.predict_a_o(XTest),yTest), "Cost on test set")
                    dnn.plot_costs()


                    yTrue, yPred = yTest, dnn.predict(XTest)
                    accuracy = accuracy_score(yTrue, yPred)
                    rocauc = roc_auc_score(yTrue, yPred)
                    
                    logreg.own_classification_report(yTest,yPred)
                    if metric=="accuracy":
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_eta = eta
                            best_lmbd = lmbd
                            best_hn = hn
                            best_act_h = act_h
                            best_rocauc = rocauc
                    elif metric=="roc_auc":
                        if rocauc > best_rocauc:
                            best_accuracy = accuracy
                            best_eta = eta
                            best_lmbd = lmbd
                            best_hn = hn
                            best_act_h = act_h
                            best_rocauc = rocauc
                        

                    else: 
                        raise ValueError("No metric chosen, exiting.")
                        


                    print("---------------------------------------------------")
                    print("Learning rate : {:<8}".format(eta), " Current best : ", best_eta) 
                    print("Lambda        : {:<8}".format(lmbd), " Current best : ", best_lmbd)
                    print("Activation    : {:<8}".format(act_h), " Current best : ", best_act_h)
                    print("Hidden neurons: {:<8}".format(hn), " Current best : ", best_hn)
                    print("roc auc       : {:.6}".format(rocauc), " Current best :  %.4f" % best_rocauc)
                    print("Accuracy      : {:.6}".format(accuracy), " Current best :  %.4f" % best_accuracy)
                    print()

    print("------------------ Best Results -------------------")
    print("Best learning rate   : ", best_eta)
    print("Best lambda          : ", best_lmbd)
    print("Best activation      : ", best_act_h)
    print("Best hidden neurons  : ", best_hn)
    print("Best rocauc          :  %.4f" % best_rocauc)
    print("Best accuracy        :  %.4f" % best_accuracy)
    print("---------------------------------------------------")
else:
    # Classify using sklearn
    clf = MLPClassifier(solver="lbfgs", alpha=1e-5,hidden_layer_sizes=(3))
    clf.fit(XTrain, yTrain)
    yTrue, yPred = yTest, clf.predict(XTest)
    print(classification_report(yTrue,yPred))
    print("Roc auc: ", roc_auc_score(yTrue,yPred))