import sys
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, classification_report, r2_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression

plt.style.use(u"~/.matplotlib/stylelib/trygveplot_astro.mplstyle")

from logreg     import LogReg
from initdata   import InitData
from DNN        import NeuralNetwork
from cost_functions import CostFunctions

"""
In this part of the project, we assess the predictive ability of a feed-forward Neural 
network on fitting a polynomial to the franke function.

To explore hyperparameter-space, chose explore = True
To run self-written NN network, chose sklearn = False
Chose metric for saving best score (R2/mse)
For credit card data, cost=mse is chosen
"""

## Get data from InitData Class ---------------------------------------
data = InitData()
noise = 0.05
XTrain, yTrain, XTest, yTest, X, y, b, x_, y_ =  data.franke_data(noise=noise, N=20)
logreg = LogReg() # init Logreg class
cf = CostFunctions()

# ----- PARAMETERS ------
sklearn = False
NN = True
explore = False
metric = "R2" # "mse"

if explore==True: # Explore parameter space for franke function
    eta_vals = np.logspace(-3, -1, 3)
    lmbd_vals =  np.logspace(-3, -1, 3)
    #acts_hidden = ["sigmoid", "tanh", "relu", "elu"]
    acts_hidden = ["logistic", "tanh", "relu"] # Supported by sklearn
    act_o = "identity" 
    hidden_neurons = [4,8,12,16,50,100] 
    epochs= 100
    tol = 0.0001
    batch_size = 1
    n_categories = 1

    
else: # Optimal setup for franke function
    act_o = "identity" 
    epochs=1000 # Max epochs
    batch_size = 1
    n_categories = 1
    tol = 0.000001
    # Best sklearn
    hidden_neurons = [50]
    #hidden_neurons = [12]
    acts_hidden = ["relu"]
    eta_vals = [1e-1]
    lmbd_vals = [1e-5]


# ------ Run network ------
def run_network(sklearn, NN, lmbd, eta, act_h, act_o, hn, epochs, tol, batch_size, n_categories, color_iter=0, length=1):
    costs = None
    if sklearn:
        if NN: # NN Regress using sklearn - Somewhat reasonable, needs tuning.
            mlpr= MLPRegressor(hidden_layer_sizes=hn,
                                activation=act_h, 
                                solver="adam", 
                                alpha = lmbd, 
                                learning_rate_init=eta
                                )
            mlpr.fit(XTrain, yTrain)

            yTrue, yPred = yTest, mlpr.predict(XTest)
            ypred = mlpr.predict(X)
            
            R2 = mlpr.score(XTest,yTrue.ravel())
            MSE = 1.0 #Not implemented

        else: # Linear regression using Sklearn  - Initial results prefer this
            mlpr = LinearRegression()
            mlpr.fit(XTrain,yTrain) 

            yTrue, yPred = yTest, mlpr.predict(XTest)
            R2 = mlpr.score(XTest,yTrue.ravel())
            MSE = 1.0
            ypred = mlpr.predict(X)
    else:
        if NN: # My regression NN - Outputs 50 :)
            mlpr = NeuralNetwork(XTrain, yTrain.reshape(-1,1),
                                XTest, yTest.reshape(-1,1),
                                eta=eta, 
                                lmbd=lmbd, 
                                n_hidden_neurons=hn,
                                act_h=act_h, 
                                act_o=act_o, 
                                epochs=epochs,
                                tol=tol,
                                batch_size=batch_size,
                                nn_type="regression",
                                cost="mse", 
                                n_categories=n_categories,
                                length=length) 
            costs, scores = mlpr.train()
                                
            # Find full data prediction
            #mlpr.plot_costs(color_iter) # Plot cost
            mlpr.plot_scores(color_iter) # Plot scores

            MSE = scores[-1,1,0]
            R2 = scores[-1,1,1]

            # full prediction
            ypred = mlpr.predict_a_o(X) 
            
        else:  # Run my regression
            raise NotImplementedError("Insert lasso n stuff")

    return MSE, R2, ypred, costs, mlpr
        
# Number of free parameters
length = len(eta_vals)*len(lmbd_vals)*len(acts_hidden)*len(hidden_neurons)

"""
OPTIMIZE NN REGRESSION
"""
best_mse = 1
best_R2  = -1e10
color_iter = 0
for i, eta in enumerate(eta_vals):
    for j, lmbd in enumerate(lmbd_vals):
        for k, act_h in enumerate(acts_hidden):
            for l, hn in enumerate(hidden_neurons):
                #hn = (hidden_neurons[l], hidden_neurons[l]) # To use multilayer for sklearn
                
                mse, R2, ypred, costs, mlpr = run_network(sklearn, NN, 
                                        lmbd=lmbd, 
                                        eta=eta,
                                        act_h=act_h,
                                        act_o=act_o, 
                                        hn=hn, 
                                        epochs=epochs,
                                        tol=tol,
                                        batch_size=batch_size,
                                        n_categories=n_categories,
                                        color_iter=color_iter,
                                        length=length)
                color_iter += 1

                #plt.loglog(mlpr.loss_curve_, label=r"{:8s} LR: {:6}   $\lambda$: {:6}  $N_h$: ({},{})  R2: {:.3f}".format(act_h, "1e"+str(int(np.log10(eta))), "1e"+str(int(np.log10(lmbd))), hn[0], hn[1], R2))
                #plt.loglog(mlpr.loss_curve_, label=r"{:8s} LR: {:6}   $\lambda$: {:6}  $N_h$: ({})  R2: {:.3f}".format(act_h, "1e"+str(int(np.log10(eta))), "1e"+str(int(np.log10(lmbd))), hn, R2))
                if metric=="mse":
                    if mse < best_mse:
                        best_mse = mse
                        best_eta = eta
                        best_lmbd = lmbd
                        best_hn = hn
                        best_act_h = act_h
                        best_ypred = ypred
                        best_mlpr = mlpr
                        best_costs = costs
                        best_R2    = R2
                elif metric=="R2":
                    if R2 > best_R2:
                        best_mse = mse
                        best_eta = eta
                        best_lmbd = lmbd
                        best_hn = hn
                        best_act_h = act_h
                        best_ypred = ypred
                        best_costs = costs
                        best_mlpr = mlpr
                        best_R2    = R2
                else: 
                    raise ValueError("No metric chosen, exiting.")
                    
                print("---------------------------------------------------")
                print("Learning rate : {:<8}".format(eta), " Current best : ", best_eta) 
                print("Lambda        : {:<8}".format(lmbd), " Current best : ", best_lmbd)
                print("Activation    : {:<8}".format(act_h), " Current best : ", best_act_h)
                print("Hidden neurons: {}".format(hn), " Current best : ", best_hn)
                print("MSE           : {:.6}".format(mse), " Current best :  %.4f" % best_mse)
                print("R2            : {:.6}".format(R2), " Current best :  %.4f" % best_R2)
                print()

#plt.loglog(best_mlpr.loss_curve_, label=r"{:8s} LR: {:6}   $\lambda$: {:6}   R2: {:.3f}".format(best_act_h, "1e"+str(int(np.log10(best_eta))), "1e"+str(int(np.log10(best_lmbd))), best_R2))
#plt.legend()
#plt.xlabel("Epoch")
#plt.ylabel("MSE loss")
#filename = "NNregression_sklearn_mlp.png"
#plt.savefig("../figs/"+filename,bbox_inches = 'tight',pad_inches = 0)
plt.show()
#sys.exit()

print("------------------ Best Results -------------------")
print("Best learning rate   : ", best_eta)
print("Best lambda          : ", best_lmbd)
print("Best activation      : ", best_act_h)
print("Best hidden neurons  : ", best_hn)
print("Best MSE             :  %.4f" % best_mse)
print("Best R2              :  %.4f" % best_R2)
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
surf = ax1.plot_surface(x_, y_, b, alpha=0.3, cmap=plt.cm.coolwarm,label=r"Franke function $N(0,%.2f)$" %noise)
surf._facecolors2d=surf._facecolors3d # Bugfix for legend
surf._edgecolors2d=surf._edgecolors3d

ax1.scatter(x_,y_,best_ypred.reshape(b.shape),alpha=1, s=1, color="C1")
filename = "NNreg_frankefit.png"
plt.savefig("../figs/"+filename,bbox_inches = 'tight',pad_inches = 0)
plt.show()

