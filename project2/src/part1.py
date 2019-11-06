import sys
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, classification_report
#plt.style.use(u"~/.matplotlib/stylelib/trygveplot_astro.mplstyle")
plt.style.use(u"../trygveplot_astro.mplstyle")

from logreg import LogReg
from initdata import InitData

"""
In this part of the project, we assess the predictive ability of logistic regression on 
determining default based on credit card data. The weights are trained using a gradient
solver and compared with Scikit-Learns Logistic regression method.
"""

## Get data from InitData Class
data = InitData()
#XTrain, yTrain, XTest, yTest, Y_train_onehot, Y_test_onehot = data.credit_data(trainingShare=0.5,per_col=True,drop_zero=True,drop_neg2=True)

#testing out dropping specific columns of the data
XTrain, yTrain, XTest, yTest, Y_train_onehot, Y_test_onehot, data_cols = data.credit_data(trainingShare=0.5,drop_zero=False,drop_neg2=False,per_col=True,return_cols=True,onehot_encode_col=['EDUCATION','MARRIAGE'])

## Initialize Logreg Class
logreg = LogReg() # init Logreg class

# Check results statistics
print("---------—--------------- True data ----------—--------—--------—")
print(" total test data: %i"%(len(yTest)))
print("               0: %i"%(len(yTest)-np.sum(yTest[:,-1])))
print("               1: %i"%(np.sum(yTest[:,-1])))
print()
      
# Optimize parameters
#lrs = np.logspace(-5,7,13)
lrs = [0.01]
for lr in lrs:
      beta, costs = logreg.SGD_batch(XTrain,yTrain.ravel(),lr=lr,adj_lr=True, rnd_seed=True, batch_size=100,n_epoch=20,verbosity=2,max_iter=10) # Fit using SGD. This can be looped over for best lambda.
      plt.plot(costs)
      print("---------—--------—--- Our Regression --------—--------—--------—")
      logreg.print_beta(cols=data_cols)
      print("-—--------—--- Training data -------—--------—")
      yPred=logreg.predict(XTrain) #predict
      logreg.own_classification_report(yTrain,yPred)
      print("-—--------—--- Validation data -------—--------—")
      yPred=logreg.predict(XTest) #predict
      logreg.own_classification_report(yTest,yPred)
      
plt.show()


# Compare with sklearn
if True: # Simple sklearn
    from sklearn.linear_model import LogisticRegression
    logReg = LogisticRegression(solver="lbfgs").fit(XTrain,yTrain.ravel())
    yTrue, yPred = yTest, logReg.predict(XTest)
    print("---------—--------—-- Sklearn Regression --------------—--------—")
    print("-—--------—--- Training data -------—--------—")
    yPred=logReg.predict(XTrain) #predict
    logreg.own_classification_report(yTrain,yPred)
    print("-—--------—--- Validation data -------—--------—")
    yPred=logReg.predict(XTest) #predict
    logreg.own_classification_report(yTest,yPred)
else:   # Fancy optimal sklearn
    logreg.sklearn_alternative(XTrain, yTrain, XTest, yTest)
