import sys
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, classification_report
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression

plt.style.use(u"~/.matplotlib/stylelib/trygveplot_astro.mplstyle")

from logreg     import LogReg
from initdata   import InitData
from DNN        import NeuralNetwork
from cost_functions import CostFunctions

## Get data from InitData Class
data = InitData()
noise = 0.1
XTrain, yTrain, XTest, yTest, X, y, b, x_, y_ =  data.franke_data(noise=noise, N=20)

logreg = LogReg() # init Logreg class

# ------ Run network ------
def run_network(sklearn, NN, lmbd, eta, act_h, act_o, hn, epochs, batch_size, n_categories):
    costs = None
    if sklearn:
        if NN: # NN Regress using sklearn - Somewhat reasonable, needs tuning.
            mlpr= MLPRegressor(hidden_layer_sizes=hn,
                                activation=act_h, 
                                solver="sgd", 
                                alpha = lmbd, 
                                learning_rate_init=eta
                                )
            mlpr.fit(XTrain, yTrain)

            yTrue, yPred = yTest, mlpr.predict(XTest)
            ypred = mlpr.predict(X)
            
            #plt.plot(mlpr.loss_curve_)
            #plt.show()

            MSE = mlpr.score(XTest,yTrue.ravel())
        else: # Linear regression using Sklearn  - Initial results prefer this
            mlpr = LinearRegression()
            mlpr.fit(XTrain,yTrain) 

            yTrue, yPred = yTest, linreg.predict(XTest)
            MSE = mlpr.score(XTest,yTrue.ravel())
            ypred = mlpr.predict(X)
    else:
        if NN: # My regression NN - Outputs 50 :)
            mlpr = NeuralNetwork(XTrain, yTrain,
                                eta=eta, 
                                lmbd=lmbd, 
                                n_hidden_neurons=hn,
                                act_h=act_h, 
                                act_o=act_o, 
                                epochs=epochs, 
                                batch_size=batch_size,
                                nn_type="regression",
                                cost="mse", 
                                n_categories=n_categories)
            costs = mlpr.train()
                                
            # Find full data prediction
            mlpr.plot_costs()

            # Test prediction
            yTrue, yPred = yTest, mlpr.predict_a_o(XTest)
            MSE = mlpr.score(yPred, yTest)
            print("lol",np.mean((yTrue-yPred)**2))

            # full prediction
            ypred = mlpr.predict_a_o(X) 
            
        else:  # Run my regression
            raise NotImplementedError("Insert lasso n stuff")

    return MSE, ypred, costs, mlpr
        
# ----- PARAMETERS ------

# Faster
sklearn = False
NN = True
explore = False

if explore==True: # Explore parameter space for franke function
    eta_vals = np.logspace(-5, 1, 7)
    lmbd_vals =  np.logspace(-5, 1, 7)
    acts_hidden = ["sigmoid", "tanh", "relu", "elu"]
    #acts_hidden = ["logistic", "tanh", "relu"] # Supported by sklearn
    act_o = "identity" 
    hidden_neurons = [4,8,12,16,50,100] 
    epochs=50
    batch_size = 1
    n_categories = 1

else: # Optimal setup for franke function
    # GOAT
    eta_vals = [0.01]
    lmbd_vals = [0.0001]
    acts_hidden = ["relu"]
    hidden_neurons = [12]
    act_o = "identity" 
    epochs=600
    batch_size = 20
    n_categories = 1


    # GOAT
    #eta_vals = [0.001]
    #lmbd_vals = [10]
    #acts_hidden = ["tanh"]
    #hidden_neurons = [4]
    
    # GOAT
    #eta_vals = [0.1]
    #lmbd_vals = [0.0001]
    #acts_hidden = ["relu"]
    #hidden_neurons = [50]


"""
OPTIMIZE NN REGRESSION
"""
best_mse = 1
for i, eta in enumerate(eta_vals):
    for j, lmbd in enumerate(lmbd_vals):
        for k, act_h in enumerate(acts_hidden):
            for l, hn in enumerate(hidden_neurons[:]):
                # If multiple layers (NO SIGNIFICANT IMPROVEMENT)
                #hn = (hidden_neurons[l+1], hidden_neurons[l]) 

                mse, ypred, costs, mlpr = run_network(sklearn, NN, 
                                        lmbd=lmbd, 
                                        eta=eta,
                                        act_h=act_h,
                                        act_o=act_o, 
                                        hn=hn, 
                                        epochs=epochs,
                                        batch_size=batch_size,
                                        n_categories=n_categories)
                if mse < best_mse:
                    best_mse = mse
                    best_eta = eta
                    best_lmbd = lmbd
                    best_hn = hn
                    best_act_h = act_h
                    best_ypred = ypred
                    best_costs = costs
                print("---------------------------------------------------")
                print("Learning rate : {:<8}".format(eta), " Current best : ", best_eta) 
                print("Lambda        : {:<8}".format(lmbd), " Current best : ", best_lmbd)
                print("Activation    : {:<8}".format(act_h), " Current best : ", best_act_h)
                print("Hidden neurons: {}".format(hn), " Current best : ", best_hn)
                print("MSE           : {:.6}".format(mse), " Current best :  %.4f" % best_mse)
                print()

print("------------------ Best Results -------------------")
print("Best learning rate   : ", best_eta)
print("Best lambda          : ", best_lmbd)
print("Best activation      : ", best_act_h)
print("Best hidden neurons  : ", best_hn)
print("Best MSE             :  %.4f" % best_mse)
print("---------------------------------------------------")



# ---- PLOT PREDICTION --------
# Plotting parameters
fig = plt.figure()
ax1 = fig.add_subplot(111, projection = '3d')
#ax1.set_zlim3d(-0.2,1.2) # Zlimit
# Remove background
ax1.xaxis.pane.fill = False
ax1.yaxis.pane.fill = False
ax1.zaxis.pane.fill = False
ax1.xaxis.pane.set_edgecolor('w')
ax1.yaxis.pane.set_edgecolor('w')
ax1.zaxis.pane.set_edgecolor('w')

ax1.view_init(5,60) # Optimal viewing angle
#surf = ax1.plot_surface(x_, y_, b, alpha=0.3, cmap=plt.cm.coolwarm,label=r"Franke function $N(0,%.2f)$" %noise)
#surf._facecolors2d=surf._facecolors3d # Bugfix for legend
#surf._edgecolors2d=surf._edgecolors3d

ax1.scatter(x_,y_,best_ypred.reshape(b.shape),alpha=1, s=1, color="C1")
plt.show()