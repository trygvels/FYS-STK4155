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
plt.style.use(u"~/.matplotlib/stylelib/trygveplot_astro.mplstyle")

# Seed parameters
seednbr = 42069
np.random.seed(seednbr)
seed(seednbr)
p = 0 # For color cycle

# Franke Function
def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

# Creating the design matrix
def design(data, degree):
    poly = PolynomialFeatures(degree=degree) # Takes a set of datapoints and calculates degree 2 polinomial
    X = poly.fit_transform(data) # Only need one dimension?
    return X

# Applying SVD to design matrix A and function b
def svd_solver(A,b):
    U, sigma, VT = la.svd(A)  # Solve SVD system
    Sigma = np.zeros(A.shape) # Initialize zeros Sigma array
    diags = np.diag(sigma) # Width of Sigma array
    hehe = len(diags)
    Sigma[:hehe,:hehe] = diags # Fill diagonal elements in top half
    Sigma_pinv = np.zeros(A.shape).T    
    Sigma_pinv[:hehe,:hehe] = np.diag(1/sigma[:hehe])
    Sigma_pinv.round(3)
    x_svd = VT.T.dot(Sigma_pinv).dot(U.T).dot(b) # Solve for x
    return x_svd

def OLS(X_train, y_train, X_test, y_test, lamb):
    if True:
        beta = svd_solver(X_train,y_train)  # Solve for x using ONLY training data
        y_train_pred = X_train.dot(beta).ravel() #reshape(y_tra.shape) # Check how beta matches training data
        y_test_pred = X_test.dot(beta).ravel() #.reshape(y_test.shape) # Test (old) x on test data
        intercept = 0
    else:
        fit = LinearRegression().fit(X_train, y_train)
        beta = fit.coef_
        intercept = fit.intercept_
        y_train_pred = fit.predict(X_train).reshape(y_train.shape)
        y_test_pred = fit.predict(X_test).reshape(y_test.shape)
    return y_train_pred, y_test_pred, beta,intercept
    
def ridge(X_train,y_train,X_test,y_test, lamb):
    if True: # Doing the algebra
        I = np.identity(X_train.shape[1])        
        beta = np.dot(np.dot(np.linalg.inv(np.dot(X_train.T, X_train) + lamb * I), X_train.T), y_train)
        y_train_pred = X_train.dot(beta).reshape(y_train.shape)
        y_test_pred = X_test.dot(beta).reshape(y_test.shape)
        intercept = 0
    else: # SKlearn for comparison
        fit = Ridge(alpha=lamb).fit(X_train, y_train)
        beta = fit.coef_
        intercept = fit.intercept_
        y_train_pred = fit.predict(X_train).reshape(y_train.shape)
        y_test_pred = fit.predict(X_test).reshape(y_test.shape)
    return y_train_pred, y_test_pred, beta, intercept

def lasso(X_train, y_train, X_test,y_test,lamb):
    fit = Lasso(alpha=lamb).fit(X_train, y_train)
    beta = fit.coef_
    intercept = fit.intercept_
    y_train_pred = fit.predict(X_train).reshape(y_train.shape)
    y_test_pred = fit.predict(X_test).reshape(y_test.shape)
    return y_train_pred, y_test_pred, beta,intercept


# Mean squared error
def mse(y, y_pred):
    mse = np.mean(np.mean((y - y_pred)**2, axis=1,keepdims=True))
    return mse

# R-squared
def R_squared(y, y_pred):
    return 1-np.sum( (y - y_pred)**2 )/np.sum( (y - np.mean(y))**2 )

# Bias
def bias(y, y_pred):
    bias = np.mean((y - np.mean(y_pred,axis=1,keepdims=True))**2)
    return  bias

# Variance 
def var2(y_pred):
    variance = np.mean(np.var(y_pred,axis=1,keepdims=True))
    return variance

# Plot surface plots
def plotterrain(y_train_pred,y_test_pred,error, besty,save,lamb):
    # Plotting parameters
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection = '3d')
    
    ax1.set_zlim3d(-0.2,1.2) # Zlimit
    #ax1.view_init(5,60) # Optimal viewing angle
    ax1.view_init(5,150) # Optimal viewing angle
    # Remove background
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False
    ax1.xaxis.pane.set_edgecolor('w')
    ax1.yaxis.pane.set_edgecolor('w')
    ax1.zaxis.pane.set_edgecolor('w')
    
    # Plot franke function #alpha=0.5
    #surf = ax1.plot_surface(x_, y_, b, alpha=0.3, cmap=cm.coolwarm,label=r"Franke function $N(0,%.2f)$" %noise)
    surf = ax1.plot_surface(x_, y_, b, alpha=0.3, cmap=cm.coolwarm,label=r"Terrain data")
    surf._facecolors2d=surf._facecolors3d # Bugfix for legend
    surf._edgecolors2d=surf._edgecolors3d
    #surf1 = ax1.plot_surface(x_, y_, besty.reshape(b.shape), alpha=.5, cmap=cm.BrBG, label="Best fit "+method.__name__+" P"+str(degree)+" $\lambda = %.4f$" %lamb)
    #surf1._facecolors2d=surf._facecolors3d # Bugfix for legend
    #surf1._edgecolors2d=surf._edgecolors3d


    # Plot training data fit
    ax1.scatter(x_,y_,besty,alpha=1, s=1, color="C1", label="Best fit "+method.__name__+" P"+str(degree)+" $\lambda$ = %.e" %lamb)
    #ax1.scatter(data_train[0],data_train[1],y_train_pred[:,-1],alpha=1, s=1, color="C1", label="Training data")
    #ax1.scatter(data_test[0],data_test[1],y_test_pred[:,-1],alpha=1, s=1, color="C0", label=r"Test data - $R^2 = %.3f$" %error)
    plt.legend()
    plt.tight_layout()
    if save==True:
        plt.savefig(str(method.__name__)+"_L"+str(lamb)+"_P"+str(degree)+"_"+datatype+".png",bbox_inches = 'tight',pad_inches = 0) # Save whatever figure youre plotting
        #plt.savefig("terrain.png",bbox_inches = 'tight',pad_inches = 0)
    plt.show()

# Print errors and plot bias-variance

def error(errors, degrees, lamb, p,alpha):
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
    plt.style.use(u"~/.matplotlib/stylelib/trygveplot_astro.mplstyle")

    if method_name=="OLS":
        lambd=""
        plt.semilogy(degrees, errors[0,1,:],label=lambd+"OLS",color="C0") # Test data
    elif method_name=="ridge":
        lambd = r"$\lambda$="+str(lamb)+" "
        plt.semilogy(degrees, errors[0,1,:],label="Ridge "+lambd,color="C1") # Test data
    elif method_name=="lasso":
        lambd = r"$\lambda$="+str(lamb)+" "
        plt.semilogy(degrees, errors[0,1,:],label="Lasso "+lambd,color="C2") # Test data




    # Error
    #plt.plot(degrees, errors[0,0,:],label=lambd+"MSE    -    Training data",color="C"+str(p),alpha=alpha) # Test data
    #plt.plot(degrees, errors[0,1,:],label=lambd+"MSE    -    Test data",color="C"+str(p),linestyle="--",alpha=alpha) # Test data
    
    ## Bias
    #plt.plot(degrees, errors[2,0,:],label=lambd+"Training data - bias",color="C"+str(p+1), linestyle=":") # Training data
    #plt.plot(degrees, errors[2,1,:],label=lambd+"Bias     -    Test data",color="C"+str(p+1), alpha=alpha) # Training data
    #
    ## Variance
    #plt.plot(degrees, errors[3,0,:],label=lambd+"Training data - variance",linestyle="--",color="C"+str(p+2))
    #plt.plot(degrees, errors[3,1,:],label=lambd+"Variance - Test data",color="C"+str(p+2),alpha=alpha) # Test data
    #
    ## Bias + Variance
    #plt.plot(degrees, errors[2,0,:]+errors[3,0,:],label=r"Training data - Variance + Bias",linestyle="-.",color="C"+str(p+3)) # Test data
    #plt.plot(degrees, errors[2,1,:]+errors[3,1,:],label=r"Test data - Variance + Bias",linestyle="-.",color="C"+str(p+3),alpha=alpha) # Test data
    
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
        
def rebin(a, shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)

if False: # Use Franke function
    # Make data.
    N = 20
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)

    x_, y_ = np.meshgrid(x,y)
    data = np.c_[(x_.ravel()).T,(y_.ravel()).T]
    data = pd.DataFrame(data)

    # Create and transform franke function data
    noise = 0.1 # Amount of noise
    b = FrankeFunction(x_, y_) + np.random.normal(size=x_.shape)*noise # Franke function with optional gaussian noise
    y = pd.DataFrame(b.ravel().T)
    datatype = "franke_N"+str(noise)
else: # Using real terrain data
    downsample = 60
    b = imread("../MachineLearning/doc/Projects/2019/Project1/Datafiles/SRTM_data_Norway_2.tif")[:-1,:-1]# [:-1:downsample,:-1:downsample]
    b = rebin(b,(60,30))
    print(b.shape)
    #b = imread("../MachineLearning/doc/Projects/2019/Project1/Datafiles/SRTM_data_Norway_2.tif")[100:120,100:120]
    b -= np.min(b)
    b = b/np.max(b)
    noise = 0.0
    length = b.shape[0]
    width = b.shape[1]
    print(length,width)
    x_,y_ = np.meshgrid(range(width), range(length))
    data = np.c_[(x_.ravel()).T,(y_.ravel()).T]
    data = pd.DataFrame(data)
    y = pd.DataFrame(b.ravel().T)
    datatype = "terrain"
# KFOLD
kfolds = 5
test_size=1/kfolds
degrees = [10] #range(15)
ylen = y.shape[0]
testsize = int(ylen/kfolds)
trainsize = ylen-testsize
alpha=1
# RESAMPLING
#kfolds = 100
#test_size=0.2
#degrees = []
#ylen = y.shape[0]
#testsize = int(ylen*test_size)
#trainsize = ylen-testsize

# Ridge gets 0.00720 on 10?
# Lasso gets 0.00824 on 28 with 1e-5


method = OLS; lambdas=[0] #00728 on 8
#method = ridge; lambdas=[0.00001];# lambdas=[0.0001]
method = lasso; lambdas=[1e-5] #0.001, 0.01,0.1,1];# lambdas=[0.0001]
#lambdas=np.logspace(-5,0,1000)
methods = [OLS, ridge, lasso] 
# Something wrong with the bias in OLS? Ridge is weird too.
def splitdata(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size, shuffle=True)
    return X_train, X_test, y_train, y_test 

def kfold(X,nsplits):
    train_idx = np.zeros((nsplits, trainsize))
    test_idx = np.zeros((nsplits, testsize))
    X=list(range(ylen))
    shuffle(X)
    for k in range(nsplits):
        X_duplicate = X.copy()
        test_idx[k,:] = X_duplicate[testsize*k:testsize*(k+1)]
        del X_duplicate[testsize*k:testsize*(k+1)]
        train_idx[k,:] = X_duplicate
    return train_idx.astype(int), test_idx.astype(int)

terms = int((degrees[0]+1)*(degrees[0]+2)/2)
bestbetas = np.zeros((terms,len(lambdas),2))
for method in methods:
    for l, lamb in enumerate(lambdas): # Iterate over all hyperparemeters lambda
        k_errors = np.zeros((4, 2, len(degrees))) # Initializing error array
        for d, degree in enumerate(degrees): # Iterate over polynomial complexity
            y_train = np.zeros((trainsize,kfolds))
            y_test = np.zeros((testsize,kfolds))
            y_train_pred = np.zeros((trainsize,kfolds))
            y_test_pred = np.zeros((testsize,kfolds))
            intercept = np.zeros(kfolds)
            terms = int((degree+1)*(degree+2)/2)
            bestbetas = np.zeros((terms,len(lambdas),2))
            betas = np.zeros((terms,kfolds))

            train_idx, test_idx = kfold(data,kfolds) # Kfold splits
            for k in range(kfolds): # K-fold test. Do it k times and check errors.
                # K FOLD STUFF
                data_train, data_test,y_tra,y_tes = data.iloc[train_idx[k,:]],\
                    data.iloc[test_idx[k,:]] ,y.iloc[train_idx[k,:]], y.iloc[test_idx[k,:]]

                # RESAMPLING (OLD)
                #data_train, data_test, y_tra, y_tes = splitdata(data,y) # Sample data

                X_train = design(data_train,degree) # Design matrix for training data
                X_test = design(data_test,degree) # Design matrix for training data
            
                y_train[:,k] = y_tra[0] # Index stuff
                y_test[:,k] = y_tes[0]

                y_train_pred[:,k], y_test_pred[:,k], betas[:,k], intercept[k] = method(X_train,y_train[:,k],\
                                                                     X_test,y_test[:,k],lamb)


            # calculate averaged over kfolds (Best fit)
            bestbetas[:,l,0] = np.mean(betas,axis=1)
            bestbetas[:,l,1] = np.var(betas,axis=1)
        
            besty = design(data, degree).dot(bestbetas[:,l,0])+np.mean(intercept)
            mse(y.values, besty)

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

            plotterrain(y_train_pred,y_test_pred, k_errors[1,1,d],besty,True, lamb) # Plot surfaces
        #error(k_errors, degrees, lamb, p,alpha) # Bias variance plot and errors
        #alpha -= 0.2
    #plt.savefig("BV_"+str(method.__name__)+"_L+"+str(lamb)+"_N"+str(noise)+"_"+datatype+".png",bbox_inches = 'tight',pad_inches = 0) # Save whatever figure youre plotting
    #plt.savefig("BV_terrain.png",bbox_inches = 'tight',pad_inches = 0) # Save whatever figure youre plotting
plt.show()    

"""
alpha = 1.0
for i in range(14):
    if i>6:
        alpha = 0.5
    plt.errorbar(lambdas, bestbetas[i,:,0],yerr=np.sqrt(bestbetas[i,:,1]), errorevery=20,label=r"$\beta_{%i}$"%i,alpha=alpha)
    plt.xscale("log")
plt.legend(loc="right")
plt.xlabel(r"$\beta$")
plt.ylabel(r"$\lambda$")
plt.tight_layout()
plt.savefig(method.__name__+"_lambdas.png",bbox_inches = 'tight',pad_inches = 0)
plt.show()
"""




