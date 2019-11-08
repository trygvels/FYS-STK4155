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
XTrain, yTrain, XTest, yTest, Y_train_onehot, Y_test_onehot, data_cols = data.credit_data(trainingShare=0.5,drop_zero=True,drop_neg2=True,per_col=True,return_cols=True,onehot_encode_col=['EDUCATION','MARRIAGE'],plt_corr=False)

## Initialize Logreg Class
logreg = LogReg(cost='cross_entropy_part1') # init Logreg class

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
      beta, costs,betas = logreg.SGD_batch(XTrain,yTrain.ravel(),lr=lr,adj_lr=True, rnd_seed=True, batch_size=100,n_epoch=50,verbosity=1,max_iter=1,new_per_iter=False) # Fit using SGD. This can be looped over for best lambda.
      plt.figure(2)
      plt.plot(costs)
      print("---------—--------—--- Our Regression --------—--------—--------—")
      logreg.print_beta(cols=data_cols,betas=betas)
      print("-—--------—--- Training data -------—--------—")
      yPred=logreg.predict(XTrain) #predict
      logreg.own_classification_report(yTrain,yPred)
      print("-—--------—--- Validation data -------—--------—")
      yPred=logreg.predict(XTest) #predict
      logreg.own_classification_report(yTest,yPred)
      plt.show()
      plt.clf
      logreg.plot_cumulative(XTest,yTest)
      logreg.print_beta_to_file(d_label=data_cols)



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
    logreg.plot_cumulative(XTest,yTest,beta=logReg.coef_.T,label='sklearn')
else:   # Fancy optimal sklearn
    logreg.sklearn_alternative(XTrain, yTrain, XTest, yTest)




"""
CURRENT OUTPUT:


-—--------—--- Training data -------—--------—
Predicting y using logreg
              precision     recall     f1-score     true number    predicted number

           0      0.878      0.806        0.841           11080               10175
           1      0.448      0.584        0.507            2980                3885

    accuracy                              0.759           14060
   macro avg      0.663      0.695        0.674           14060
weighted avg      0.787      0.759        0.770           14060

-—--------—--- Validation data -------—--------—
Predicting y using logreg
              precision     recall     f1-score     true number    predicted number

           0      0.873      0.801        0.836           10998               10090
           1      0.450      0.583        0.508            3063                3971

    accuracy                              0.754           14061
   macro avg      0.662      0.692        0.672           14061
weighted avg      0.781      0.754        0.764           14061

---------—--------—-- Sklearn Regression --------------—--------—
-—--------—--- Training data -------—--------—
              precision     recall     f1-score     true number    predicted number

           0      0.846      0.949        0.895           11080               12428
           1      0.657      0.360        0.465            2980                1632

    accuracy                              0.824           14060
   macro avg      0.752      0.655        0.680           14060
weighted avg      0.806      0.824        0.804           14060

-—--------—--- Validation data -------—--------—
              precision     recall     f1-score     true number    predicted number

           0      0.842      0.952        0.894           10998               12430
           1      0.674      0.359        0.469            3063                1631

    accuracy                              0.823           14061
   macro avg      0.758      0.655        0.681           14061
weighted avg      0.806      0.823        0.801           14061
"""
