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

# unit test (exits afterwards)
if (False):
      logreg = LogReg(cost='cross_entropy') # init Logreg class
      logreg.unit_test() # see logreg for expected output

## Get data from InitData Class
data = InitData()

##initialize all data, splitting gender, education and marital status
#XTrain, yTrain, XTest, yTest, Y_train_onehot, Y_test_onehot = data.credit_data(trainingShare=0.5,per_col=True,drop_zero=True,drop_neg2=True)

##initialize all data (with some bill_amt and pay_amt), split education and marital status
XTrain, yTrain, XTest, yTest, Y_train_onehot, Y_test_onehot, data_cols = data.credit_data(trainingShare=0.5,drop_zero=False,drop_neg2=False,per_col=True,return_cols=True,onehot_encode_col=['EDUCATION','MARRIAGE'],plt_corr=False,plot_alldata=False)

##Initialize only data without '0' and '-2' in payment history
#XTrain, yTrain, XTest, yTest, Y_train_onehot, Y_test_onehot, data_cols = data.credit_data(trainingShare=0.5,drop_zero=True,drop_neg2=True,per_col=True,return_cols=True,onehot_encode_col=['EDUCATION','MARRIAGE'],plt_corr=False,plot_alldata=False)

## Initialize Logreg Class
logreg = LogReg(cost='cross_entropy') # init Logreg class

# Check results statistics
print("---------—--------------- True data ----------—--------—--------—")
print(" total test data: %i"%(len(yTest)))
print("               0: %i"%(len(yTest)-np.sum(yTest[:,-1])))
print("               1: %i"%(np.sum(yTest[:,-1])))
print()

# Optimize parameters
#lrs = np.logspace(-5,7,13)

niter=50
sgd=False
plt_cost=False

if (sgd):
      if plt_cost:
            lrs = [0.1,0.01,0.001]
      else:
            lrs = [0.001]
else:
      if plt_cost:
            lrs = [1.0,0.5,0.1,0.01]
      else:
            lrs = [0.1]

return_ar=True
f1_log=[]
f3_log=[]
ar_log=[]
ac_log=[]
ac1_log=[]
if plt_cost:
      plt.figure(1,figsize=(7,7))
      niter=len(lrs)

for i in range(niter):
      if plt_cost:
            j=i
      else:
            j=0
            
      print('%i of %i'%(i+1,niter))
      if (sgd):
            beta, costs,betas = logreg.SGD_batch(XTrain,yTrain.ravel(),lr=lrs[j],adj_lr=True, rnd_seed=True, batch_size=100,n_epoch=25,verbosity=1,n_iter=10,new_per_iter=False) # Fit using SGD. This can be looped over for best lambda (i.e. learning rate 'lr').
      else:
            beta, costs = logreg.GD(XTrain,yTrain.ravel(),lr=lrs[j], rnd_seed=True,tol=1e-2) # Fit using GD. This can be looped over for best lambda (i.e. learning rate 'lr').
            betas=beta.copy()
      if plt_cost:
            plt.plot(costs,label='%5.3f'%lrs[i])
      yPred=logreg.predict(XTest) #predict
      f1,ac1=logreg.own_classification_report(yTest,yPred,return_f1=True,return_ac=True)
      yPred=logreg.predict(XTrain) #predict
      f2,ac2=logreg.own_classification_report(yTrain,yPred,return_f1=True,return_ac=True)
      f3=(f1+f2)/2.0
      ac=(ac1+ac2)/2.0
      f1_log.append(f1)
      f3_log.append(f3)
      ac_log.append(ac)
      ac1_log.append(ac1)
      if (return_ar):
            ar=logreg.plot_cumulative(XTest,yTest,return_ar=return_ar)
            ar_log.append(ar)
      else:
            logreg.plot_cumulative(XTest,yTest,return_ar=return_ar)
            logreg.print_beta_to_file(d_label=data_cols)
            ar=0
      if (i==0):
            ar_best=ar
            f1_best=f1
            f3_best=f3
            f1_beta=beta.copy()
            f3_beta=beta.copy()
            ar_beta=beta.copy()
      else:
            if (ar>ar_best):
                  ar_best=ar
                  ar_beta=beta.copy()
            if (f1>f1_best):
                  f1_best=f1
                  f1_beta=beta.copy()
            if (f3>f3_best):
                  f3_best=f3
                  f3_beta=beta.copy()

if plt_cost:
      if (not sgd):
            plt.xscale('log')
            plt.yticks(fontsize=16)
            plt.xlabel('Step number',fontsize=20)
      else:
            plt.yscale('log')
            plt.yticks([2000,3000,5000,10000],['2000','3000','5000','10000'],fontsize=16)
            plt.xlabel('Epoch number',fontsize=20)
      plt.ylabel('Cost',fontsize=20)
      plt.xticks(fontsize=16)
      plt.legend(loc='upper right',fontsize=16)
      plt.savefig('../figs/cost.pdf',bbox_inches='tight',pad_inches=0.02)
      plt.show()
      exit()
                  
print("---------—--------—--- Our Regression --------—--------—--------—")

f1_log=np.array(f1_log)
f3_log=np.array(f3_log)
ac_log=np.array(ac_log)
ac1_log=np.array(ac1_log)
print('label    mean    std   best')
if (return_ar):
      ar_log=np.array(ar_log)
      print('%5s   %6.4f  %6.4f   %6.4f'%('ar',ar_log.mean(),ar_log.std(),ar_log.max()))
print('%5s   %6.4f  %6.4f   %6.4f'%('f1',f1_log.mean(),f1_log.std(),f1_log.max()))
print('%5s   %6.4f  %6.4f   %6.4f'%('f3',f3_log.mean(),f3_log.std(),f3_log.max()))
print('%5s   %6.4f  %6.4f   %6.4f'%('ac1',ac1_log.mean(),ac1_log.std(),ac1_log.max()))
print('%5s   %6.4f  %6.4f   %6.4f'%('ac_avg',ac_log.mean(),ac_log.std(),ac_log.max()))
print()


if (return_ar):
      yPred=logreg.predict(XTest,betas=ar_beta)
      f1=logreg.own_classification_report(yTest,yPred,return_f1=True)
      yPred=logreg.predict(XTrain,betas=ar_beta) #predict
      f2=logreg.own_classification_report(yTrain,yPred,return_f1=True)
      f3=(f1+f2)/2.0
      if (XTest.shape[0]>3000):
            label='_best_ar_%5.3f_f1_%5.3f_f3_%5.3f_all'%(ar_best,f1,f3)
      else:
            label='_best_ar_%5.3f_f1_%5.3f_f3_%5.3f'%(ar_best,f1,f3)
      if (sgd):
            label = 'SGD'+label
      else:
            label = 'GD'+label
      logreg.plot_cumulative(XTest,yTest,beta=ar_beta,label=label)
      logreg.print_beta_to_file(d_label=data_cols,beta=ar_beta,label=label)
      print()
      print('Best beta, given area ratio value')
      logreg.print_beta(cols=data_cols,betas=ar_beta)
      print("-—--------—--- Training data -------—--------—")
      yPred=logreg.predict(XTrain,betas=ar_beta) #predict
      logreg.own_classification_report(yTrain,yPred)
      print("-—--------—--- Validation data -------—--------—")
      yPred=logreg.predict(XTest,betas=ar_beta) #predict
      logreg.own_classification_report(yTest,yPred)
      ar_log=np.array(ar_log)


ar=logreg.plot_cumulative(XTest,yTest,beta=f1_beta,return_ar=True)
yPred=logreg.predict(XTrain,betas=f1_beta) #predict
f2=logreg.own_classification_report(yTrain,yPred,return_f1=True)
f3=(f1_best+f2)/2.0
if (XTest.shape[0]>3000):
      label='_best_f1_%5.3f_f3_%5.3f_ar_%5.3f_all'%(f1_best,f3,ar)
else:
      label='_best_f1_%5.3f_f3_%5.3f_ar_%5.3f'%(f1_best,f3,ar)
if (sgd):
      label = 'SGD'+label
else:
      label = 'GD'+label
logreg.plot_cumulative(XTest,yTest,beta=f1_beta,label=label)
logreg.print_beta_to_file(d_label=data_cols,beta=f1_beta,label=label)
print()
print('Best beta, given F1 value of Test data')
logreg.print_beta(cols=data_cols,betas=f1_beta)
print("-—--------—--- Training data -------—--------—")
yPred=logreg.predict(XTrain,betas=f1_beta) #predict
logreg.own_classification_report(yTrain,yPred)
print("-—--------—--- Validation data -------—--------—")
yPred=logreg.predict(XTest,betas=f1_beta) #predict
logreg.own_classification_report(yTest,yPred)

ar=logreg.plot_cumulative(XTest,yTest,beta=f3_beta,return_ar=True)
yPred=logreg.predict(XTest,betas=f3_beta) #predict
f1=logreg.own_classification_report(yTest,yPred,return_f1=True)
yPred=logreg.predict(XTrain,betas=f3_beta) #predict
f2=logreg.own_classification_report(yTrain,yPred,return_f1=True)
f3=(f1+f2)/2.0
if (XTest.shape[0]>3000):
      label='_best_f3_%5.3f_f1_%5.3f_f2_%5.3f_ar_%5.3f_all'%(f3,f1,f2,ar)
else:
      label='_best_f3_%5.3f_f1_%5.3f_f2_%5.3f_ar_%5.3f'%(f3,f1,f2,ar)
if (sgd):
      label = 'SGD'+label
else:
      label = 'GD'+label
logreg.plot_cumulative(XTest,yTest,beta=f1_beta,label=label)
logreg.print_beta_to_file(d_label=data_cols,beta=f1_beta,label=label)
print()
print('Best beta, given avg F1 value Train/Test')
logreg.print_beta(cols=data_cols,betas=f3_beta)
print("-—--------—--- Training data -------—--------—")
yPred=logreg.predict(XTrain,betas=f3_beta) #predict
logreg.own_classification_report(yTrain,yPred)
print("-—--------—--- Validation data -------—--------—")
yPred=logreg.predict(XTest,betas=f3_beta) #predict
logreg.own_classification_report(yTest,yPred)

# Compare with sklearn
if True: # Simple sklearn
    from sklearn.linear_model import LogisticRegression
    logReg = LogisticRegression(solver="lbfgs",max_iter=1000).fit(XTrain,yTrain.ravel())
    yTrue, yPred = yTest, logReg.predict(XTest)
    print("---------—--------—-- Sklearn Regression --------------—--------—")
    print("-—--------—--- Training data -------—--------—")
    yPred=logReg.predict(XTrain) #predict
    f2,ac2=logreg.own_classification_report(yTrain,yPred,return_f1=True,return_ac=True)
    logreg.own_classification_report(yTrain,yPred)
    print("-—--------—--- Validation data -------—--------—")
    yPred=logReg.predict(XTest) #predict
    f1,ac1=logreg.own_classification_report(yTest,yPred,return_f1=True,return_ac=True)
    logreg.own_classification_report(yTest,yPred)
    print(f1,f2,ac1,ac2)
    f3=(f1+f2)/2
    ac3=(ac1+ac2)/2
    if (XTest.shape[0]>3000):
          label='sklearn_all'
    else:
          label='sklearn'
    ar=logreg.plot_cumulative(XTest,yTest,beta=logReg.coef_.T,label=label,return_ar=True)
    print('ar',ar)
    print('f1',f1)
    print('f3',f3)
    print('ac1',ac1)
    print('ac3',ac3)
    print()
    logreg.plot_cumulative(XTest,yTest,beta=logReg.coef_.T,label='lift',plt_ar=False)
else:   # Fancy optimal sklearn
    logreg.sklearn_alternative(XTrain, yTrain, XTest, yTest)

print('label    mean    std    best')
if (return_ar):
      print('%5s   %6.4f  %6.4f   %6.4f'%('ar',ar_log.mean(),ar_log.std(),ar_log.max()))
print('%5s   %6.4f  %6.4f   %6.4f'%('f1',f1_log.mean(),f1_log.std(),f1_log.max()))
print('%5s   %6.4f  %6.4f   %6.4f'%('f3',f3_log.mean(),f3_log.std(),f3_log.max()))
print('%5s   %6.4f  %6.4f   %6.4f'%('ac1',ac1_log.mean(),ac1_log.std(),ac1_log.max()))
print('%5s   %6.4f  %6.4f   %6.4f'%('ac_avg',ac_log.mean(),ac_log.std(),ac_log.max()))
print()



"""
CURRENT OUTPUT OF :


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
