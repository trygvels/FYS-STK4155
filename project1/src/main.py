import sys
from mpl_toolkits.mplot3d import Axes3D
from imageio import imread
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import numpy.linalg as la
from random import random, seed, shuffle
import pandas as pd
import pytest
from functions import *
from paramfile import params
plt.style.use(u"~/.matplotlib/stylelib/trygveplot_astro.mplstyle")

# Seed parameters
seednbr = 42069
np.random.seed(seednbr)
seed(seednbr)
p = 0 # For color cycle

# Get parameters for run
dataset, noise, degrees, methods, lambdas, plotterrain_, ploterror_, plotbetas_, kfolds, test = params()

# Plot surface plots
def plotterrain(y_train_pred,y_test_pred,error, besty,save,lamb, dataset):
    # Plotting parameters
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection = '3d')
    ax1.set_zlim3d(-0.2,1.2) # Zlimit
    # Remove background
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False
    ax1.xaxis.pane.set_edgecolor('w')
    ax1.yaxis.pane.set_edgecolor('w')
    ax1.zaxis.pane.set_edgecolor('w')

    
    if dataset.casefold()==("franke").casefold():
        ax1.view_init(5,60) # Optimal viewing angle
        surf = ax1.plot_surface(x_, y_, b, alpha=0.3, cmap=cm.coolwarm,label=r"Franke function $N(0,%.2f)$" %noise)
    else:
        ax1.view_init(5,150) # Optimal viewing angle
        surf = ax1.plot_surface(x_, y_, b, alpha=0.3, cmap=cm.coolwarm,label=r"Terrain data")        

    # Plot franke function #alpha=0.5
    surf._facecolors2d=surf._facecolors3d # Bugfix for legend
    surf._edgecolors2d=surf._edgecolors3d

    # Plot prediction as surface
    #surf1 = ax1.plot_surface(x_, y_, besty.reshape(b.shape), alpha=.5, cmap=cm.BrBG, label="Best fit "+method.__name__+" P"+str(degree)+" $\lambda = %.4f$" %lamb)
    #surf1._facecolors2d=surf._facecolors3d # Bugfix for legend
    #surf1._edgecolors2d=surf._edgecolors3d


    # Plot training data fit
    ax1.scatter(x_,y_,besty,alpha=1, s=1, color="C1", label="Best fit "+method.__name__+" P"+str(degree)+" $\lambda$ = %.e" %lamb)
    # Plot train and test data separately
    #ax1.scatter(data_train[0],data_train[1],y_train_pred[:,-1],alpha=1, s=1, color="C1", label="Training data")
    #ax1.scatter(data_test[0],data_test[1],y_test_pred[:,-1],alpha=1, s=1, color="C0", label=r"Test data - $R^2 = %.3f$" %error)

    plt.legend()
    plt.tight_layout()
    if save==True:
        plt.savefig(str(method.__name__)+"_L"+str(lamb)+"_P"+str(degree)+"_"+datatype+".png",bbox_inches = 'tight',pad_inches = 0) # Save whatever figure youre plotting
        #plt.savefig("terrain.png",bbox_inches = 'tight',pad_inches = 0)
    plt.show()


def error(errors, degrees, lamb, p,alpha): # Print errors and plot bias-variance
    for i, degree in enumerate(degrees):
        print(" ")
        print("Polynomial of degree: ", degree)
        print("Total error on training data: ")
        print("MSE: %.5f,  R^2: %.5f" %(errors[0,0,i], errors[1,0,i]))
        
        print("Total error on test data: ")
        print("MSE: %.5f,  R^2: %.5f" %(errors[0,1,i], errors[1,1,i]))
                
    if len(degrees)<2:
        print('Too few polynomial degrees for bias-variance plot')
        return False
    
    method_name = str(method.__name__)

    if method_name=="OLS":
        lambd=""
        plt.plot(degrees, errors[0,1,:],label=lambd+"OLS",color="C0") # Test data
    elif method_name=="ridge":
        lambd = r"$\lambda$="+str(lamb)+" "
        plt.plot(degrees, errors[0,1,:],label="Ridge "+lambd,color="C1") # Test data
    elif method_name=="lasso":
        lambd = r"$\lambda$="+str(lamb)+" "
        plt.plot(degrees, errors[0,1,:],label="Lasso "+lambd,color="C2") # Test data


    # Error
    #plt.plot(degrees, errors[0,0,:],label=lambd+"MSE    -    Training data",color="C"+str(p),alpha=alpha) # Test data
    #plt.plot(degrees, errors[0,1,:],label=lambd+"MSE    -    Test data",color="C"+str(p),linestyle="--",alpha=alpha) # Test data
    
    ## Bias
    #plt.plot(degrees, errors[2,0,:],label=lambd+"Training data - bias",color="C"+str(p+1), linestyle=":") # Training data
    plt.plot(degrees, errors[2,1,:],label=lambd+"Bias     -    Test data",color="C"+str(p+1), alpha=alpha) # Training data
    #
    ## Variance
    #plt.plot(degrees, errors[3,0,:],label=lambd+"Training data - variance",linestyle="--",color="C"+str(p+2))
    plt.plot(degrees, errors[3,1,:],label=lambd+"Variance - Test data",color="C"+str(p+2),alpha=alpha) # Test data
   
    p += 1 #Color cycle for each lambda
    #plt.title("MSE for training and test data")
    plt.xlabel("Polynomial degree")
    plt.ylabel("Prediction error")
    #plt.title(method_name)
    plt.xlim([0,len(degrees)-1])
    plt.ylim([0,errors[0,0,0]*1.5])
    plt.legend()
    #if alpha==1:
    #    plt.legend()


if dataset=="franke" or dataset=="Franke": # Use Franke function
    # Make data.
    N = 20
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)

    x_, y_ = np.meshgrid(x,y)
    data = np.c_[(x_.ravel()).T,(y_.ravel()).T]
    data = pd.DataFrame(data)

    # Create and transform franke function data
    b = FrankeFunction(x_, y_) + np.random.normal(size=x_.shape)*noise # Franke function with optional gaussian noise
    y = pd.DataFrame(b.ravel().T)
    datatype = "franke_N"+str(noise)
elif dataset=="terrain" or dataset=="Terrain": # Using real terrain data
    b = imread("../../MachineLearning/doc/Projects/2019/Project1/Datafiles/SRTM_data_Norway_2.tif")[:-1,:-1]
    b = rebin(b,(60,30))
    print(b.shape)
    b -= np.min(b)
    b = b/np.max(b)
    length = b.shape[0]
    width = b.shape[1]
    x_,y_ = np.meshgrid(range(width), range(length))
    data = np.c_[(x_.ravel()).T,(y_.ravel()).T]
    data = pd.DataFrame(data)
    y = pd.DataFrame(b.ravel().T)
    datatype = "terrain"
else:
    print("no dataset specified")
    sys.exit()

# KFOLD
test_size=1/kfolds
ylen = y.shape[0]
testsize = int(ylen/kfolds)
trainsize = ylen-testsize
alpha=1 # Start alpha for plotting with decreasing transparency
terms = int((degrees[0]+1)*(degrees[0]+2)/2) # Number of beta terms per polynomial degrees
bestbetas = np.zeros((terms,len(lambdas),2)) # Array with best beta values after k-fold
def runthething(test):
    global terms, method, degree
    for method in methods:
        bestbetas     = np.zeros((terms,len(lambdas),2))
        for l, lamb in enumerate(lambdas): # Iterate over all hyperparemeters lambda
            k_errors = np.zeros((4, 2, len(degrees))) # Initializing error array
            for d, degree in enumerate(degrees): # Iterate over polynomial complexity
            
                # Create arrays 
                y_train       = np.zeros((trainsize,kfolds))
                y_test        = np.zeros((testsize,kfolds))
                y_train_pred  = np.zeros((trainsize,kfolds))
                y_test_pred   = np.zeros((testsize,kfolds))
                intercept     = np.zeros(kfolds)
                terms         = int((degree+1)*(degree+2)/2)
                betas         = np.zeros((terms,kfolds))

                if len(degrees)>1: # Fix bug with varying number of terms
                    bestbetas     = np.zeros((terms,len(lambdas),2))
            
                train_idx, test_idx = kfold(data,kfolds,testsize,trainsize, ylen) # Kfold splits
            
                for k in range(kfolds): # K-fold test. Do it k times and check errors.

                    # Get test and train data for this fold
                    data_train, data_test,y_tra,y_tes = data.iloc[train_idx[k,:]],\
                        data.iloc[test_idx[k,:]] ,y.iloc[train_idx[k,:]], y.iloc[test_idx[k,:]]

                    # OLD RESAMPLING
                    #data_train, data_test, y_tra, y_tes = splitdata(data,y) # Sample data

                    X_train = design(data_train,degree) # Design matrix for training data
                    X_test  = design(data_test,degree)  # Design matrix for training data
            
                    y_train[:,k] = y_tra[0] # Index stuff
                    y_test[:,k] = y_tes[0]

                    y_train_pred[:,k], y_test_pred[:,k], betas[:,k], intercept[k] = method(X_train,y_train[:,k],\
                                                                                       X_test,y_test[:,k],lamb, test)

                # calculate averaged over kfolds (Best fit)
                # Should be weighted
                bestbetas[:,l,0] = np.mean(betas,axis=1)
                bestbetas[:,l,1] = np.var(betas,axis=1) # Variance within folds
        
                besty = design(data, degree).dot(bestbetas[:,l,0])+np.mean(intercept) # Calculate y on full data
                mse(y.values, besty) # Error on full data

                # Save training errors
                k_errors[0, 0, d] = mse(y_train, y_train_pred)
                k_errors[1, 0, d] = R_squared(y_train, y_train_pred)
                k_errors[2, 0, d] = bias(y_train, y_train_pred)
                k_errors[3, 0, d] = var2(y_train_pred)
            
                # Save test errors
                k_errors[0, 1, d] = mse(y_test, y_test_pred)
                k_errors[1, 1, d] = R_squared(y_test, y_test_pred)
                k_errors[2, 1, d] = bias(y_test, y_test_pred)
                k_errors[3, 1, d] = var2(y_test_pred)

                if plotterrain_:
                    plotterrain(y_train_pred, y_test_pred, k_errors[1,1,d], besty, False, lamb,dataset) # Plot surfaces
            
            alpha = 1 # Used for decreasing plot alpha per iteration
            if ploterror_:
                error(k_errors, degrees, lamb, p,alpha) # Bias variance plot and errors
        #plt.savefig("BV_"+str(method.__name__)+"_L+"+str(lamb)+"_N"+str(noise)+"_"+datatype+".png",bbox_inches = 'tight',pad_inches = 0) # Save whatever figure youre plotting
        #plt.savefig("BV_terrain.png",bbox_inches = 'tight',pad_inches = 0) # Save whatever figure youre plotting
        plt.show()    

        if plotbetas_:     # Plotting betas per lambda
            alpha = 1.0
            for i in range(14):
                if i>6:
                    alpha = 0.5
                plt.errorbar(lambdas, bestbetas[i,:,0],yerr=np.sqrt(bestbetas[i,:,1]), errorevery=20,label=r"$\beta_{%i}$"%i,alpha=alpha)
                plt.xscale("log")
            plt.legend(loc="right")
            plt.ylabel(r"$\beta$")
            plt.xlabel(r"$\lambda$")
            plt.tight_layout()
            #plt.savefig(method.__name__+"_lambdas.png",bbox_inches = 'tight',pad_inches = 0)
            plt.show()
        return besty, k_errors[3,1,-1]

# If test=true, then check against sklearn values    
if test:
    besty_homemade, variance = runthething(test=False)
    besty_sklearn, variance = runthething(test)
    print("variance:",variance)
    print("Min and max values of sklearn: ", min(besty_sklearn), max(besty_sklearn))
    print("Max absolute difference: ", max(abs(besty_homemade-besty_sklearn)))

    # Plot difference plot
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection = '3d')
    # Remove background
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False
    ax1.xaxis.pane.set_edgecolor('w')
    ax1.yaxis.pane.set_edgecolor('w')
    ax1.zaxis.pane.set_edgecolor('w')

    diff=besty_homemade-besty_sklearn
    surf = ax1.plot_surface(x_, y_, diff.reshape(b.shape), alpha=0.9, cmap=cm.coolwarm)
    plt.tight_layout()
    plt.show()

    assert besty_homemade == pytest.approx(besty_sklearn, abs=variance), "The difference between self-implemented method and scikit-learn is not within variance level (%.4f)" %variance
    print("Self implementation of the method is reasonably close to the SKlearn alternative.")
else:
    besty,variance = runthething(test)
    
    





