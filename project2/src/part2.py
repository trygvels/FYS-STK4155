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

------------------ Best Results -------------------
Best lambda          :  0.0001
Best learning rate   :  0.01
Best activation      :  4
Best hidden neurons  :  sigmoid
Best accuracy        :  0.8236
Full data

---------------------------------------------------
Learning rate : 0.0001    Current best :  0.0001
Lambda        : 1.0       Current best :  0.1
Activation    : sigmoid   Current best :  relu
Hidden neurons: 50        Current best :  16
Accuracy      : 0.808335  Current best :  0.8258
"""
## Params
explore = False
sklearn = True


## Get data from InitData Class
data = InitData()
XTrain, yTrain, XTest, yTest, Y_train_onehot, Y_test_onehot =  data.credit_data(trainingShare=0.5)#, drop_zero=True,drop_neg2=True)


logreg = LogReg() # init Logreg class


if explore=True:
    eta_vals = np.logspace(-5, 1, 7)
    lmbd_vals = np.logspace(-5, 1, 7)
    activations = ["sigmoid", "tanh", "relu", "elu"]
    hidden_neurons = [4,8,12,16,50,100] 
else:
    eta_vals = [0.01]
    lmbd_vals = [0.0001]
    activations = ["sigmoid"]
    hidden_neurons = [4] 

if sklearn = False:
    # grid search
    best_accuracy = 0
    for i, eta in enumerate(eta_vals):
        for j, lmbd in enumerate(lmbd_vals):
            for k, act in enumerate(activations):
                for l, hn in enumerate(hidden_neurons):
                    dnn = NeuralNetwork(XTrain, Y_train_onehot.toarray(), eta=eta, lmbd=lmbd, n_hidden_neurons=hn, activation=act)
                    dnn.train()
                    
                    
                    test_predict = dnn.predict(XTest)
                    

                    accuracy = accuracy_score(yTest, test_predict)
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_eta = eta
                        best_lmbd = lmbd
                        best_hn = hn
                        best_act = act

                    print("---------------------------------------------------")
                    print("Learning rate : {:<8}".format(eta), " Current best : ", best_eta) 
                    print("Lambda        : {:<8}".format(lmbd), " Current best : ", best_lmbd)
                    print("Activation    : {:<8}".format(act), " Current best : ", best_act)
                    print("Hidden neurons: {:<8}".format(hn), " Current best : ", best_hn)
                    print("Accuracy      : {:.6}".format(accuracy), " Current best :  %.4f" % best_accuracy)
                    print()

    print("------------------ Best Results -------------------")
    print("Best lambda          : ", best_lmbd)
    print("Best learning rate   : ", best_eta)
    print("Best activation      : ", best_act)
    print("Best hidden neurons  : ", best_hn)
    print("Best accuracy        :  %.4f" % best_accuracy)
    print("---------------------------------------------------")

else:
    # Classify using sklearn
    clf = MLPClassifier(solver="lbfgs", alpha=1e-5,hidden_layer_sizes=(3))
    clf.fit(XTrain, yTrain)
    yTrue, yPred = yTest, clf.predict(XTest)
    logreg.own_classification_report(yTrain,yPred)
