import sys
from mpl_toolkits.mplot3d import Axes3D
from imageio import imread
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LinearRegression
import numpy as np
import numpy.linalg as la
from random import random, seed
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# Seed parameters
seed = 42069
np.random.seed(seed)
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
    Sigma = np.zeros(A.shape) # Initialize empty Sigma array
    diags = np.diag(sigma) # Width of Sigma array
    hehe = len(diags)
    Sigma[:hehe,:hehe] = diags # Fill diagonal elements in top half
    Sigma_pinv = np.zeros(A.shape).T    
    Sigma_pinv[:hehe,:hehe] = np.diag(1/sigma[:hehe])
    Sigma_pinv.round(3)

    x_svd = VT.T.dot(Sigma_pinv).dot(U.T).dot(b) # Solve for x
    return x_svd

def OLS(X_train, y_train, X_test, y_test, lamb):
    beta = svd_solver(X_train,y_train)  # Solve for x using ONLY training data
    y_train_pred = X_train.dot(beta).ravel() #reshape(y_tra.shape) # Check how beta matches training data
    
    y_test_pred = X_test.dot(beta).ravel() #.reshape(y_test.shape) # Test (old) x on test data
    #fit = LinearRegression().fit(X_train, y_train)
    #y_train_pred = fit.predict(X_train).reshape(y_train.shape)
    return y_train_pred, y_test_pred, beta
    
def ridge(X_train,y_train,X_test,y_test, lamb):
    X_train_reduced = X_train[0] + X_train - np.mean(X_train, axis=0) # Zero center training data
    y_train_reduced = y_train - np.mean(y_train)

    beta = (1-lamb)**-1*svd_solver(X_train_reduced, y_train_reduced) 

    y_train_pred = X_train_reduced.dot(beta).reshape(y_train_reduced.shape) # Check how beta matches training data                
    y_train_pred += np.mean(y_train)

    #fit = Ridge(alpha=lamb).fit(X_train, y_train)
    #y_train_pred = fit.predict(X_train).reshape(y_train.shape)

    X_test_reduced = X_test - np.mean(X_train, axis=0)
    y_test_pred = X_test_reduced.dot(beta).reshape(y_test.shape) # Test (old) x on test data                  

    #y_test_pred = fit.predict(X_test).reshape(y_test.shape)
    y_test_pred += np.mean(y_train)
    return y_train_pred, y_test_pred, beta

def lasso(X_train, y_train, X_test,y_test,lamb):
    fit = Lasso(alpha=lamb).fit(X_train, y_train)
    beta = fit.coef_
    y_train_pred = fit.predict(X_train).reshape(y_train.shape)
    y_test_pred = fit.predict(X_test).reshape(y_test.shape)
    return y_train_pred, y_test_pred, beta


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

# Optional plotting step
def plotstuff(y_train_pred,y_test_pred):
    # Plotting parameters
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection = '3d')
    #ax1.set_zlim3d(-0.2,1.2)
    #ax1.set_zlim3d(-100,400)
    
    # Plot franke function
    surf = ax1.plot_surface(x_, y_, b, alpha=0.5, cmap=cm.coolwarm,label="Franke function") # Plot franke function
    surf._facecolors2d=surf._facecolors3d # Bugfix for legend
    surf._edgecolors2d=surf._edgecolors3d
    
    # Plot training data fit
    ax1.scatter(data_train[0],data_train[1],y_train_pred,alpha=1, s=1, color="C0", label="Training data")
    ax1.scatter(data_test[0],data_test[1],y_test_pred,alpha=1, s=1, color="C1", label="Test data")
    
    plt.legend()
    plt.show()


def error(errors,degrees,filename,lamb,p,printerror):
    if printerror==True:
        for i, degree in enumerate(degrees):
            
            print(" ")
            print("Polynomial of degree: ", i)
            print("Total error on training data: ")
            print("MSE: %.5f,  R^2: %.5f" %(errors[0,0,i], errors[1,0,i]))

            print("Total error on test data: ")
            print("MSE: %.5f,  R^2: %.5f" %(errors[0,1,i], errors[1,1,i]))
                
    if len(degrees)<2:
        raise ValueError('Too few polynomial degrees for bias-variance plot')
    method_name = str(method.__name__)

    if method_name!="OLS":
        lambd = r"$\lambda$="+str(lamb)+" "
    else:
        lambd=""

    plt.style.use(u"~/.matplotlib/stylelib/trygveplot_astro.mplstyle")
    alpha = 0.5

    # Error
    plt.plot(degrees, errors[0,0,:],label=lambd+"Training data MSE",color="C"+str(p)) # Test data
    plt.plot(degrees, errors[0,1,:],label=lambd+"Test data MSE",color="C"+str(p),alpha=alpha) # Test data
    
    # Bias
    plt.plot(degrees, errors[2,0,:],label=lambd+"Training data - bias",color="C"+str(p+1), linestyle=":") # Training data
    plt.plot(degrees, errors[2,1,:],label=lambd+"Test data - bias",color="C"+str(p+1), linestyle=":", alpha=alpha) # Training data
    
    # Variance
    plt.plot(degrees, errors[3,0,:],label=lambd+"Training data - variance",linestyle="--",color="C"+str(p+2))
    plt.plot(degrees, errors[3,1,:],label=lambd+"Test data - variance",linestyle="--",color="C"+str(p+2),alpha=alpha) # Test data
    
    # Bias + Variance
    plt.plot(degrees, errors[2,0,:]+errors[3,0,:],label=r"Training data - Variance + Bias",linestyle="-.",color="C"+str(p+3)) # Test data
    plt.plot(degrees, errors[2,1,:]+errors[3,1,:],label=r"Test data - Variance + Bias",linestyle="-.",color="C"+str(p+3),alpha=alpha) # Test data
    
    p += 1 #Color cycle for each lambda
    #plt.title("MSE for training and test data")
    plt.xlabel("Polynomial degree")
    plt.ylabel("Prediction error")
    plt.title(method_name)
    plt.xlim([0,len(degrees)])
    #plt.ylim([0,0.1])
    plt.legend()

if True:
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
else:
    b = imread("../MachineLearning/doc/Projects/2019/Project1/Datafiles/SRTM_data_Norway_2.tif")[100:120,100:120]
    length = b.shape[0]
    width = b.shape[1]
    x_,y_ = np.meshgrid(range(width), range(length))
    data = np.c_[(x_.ravel()).T,(y_.ravel()).T]
    data = pd.DataFrame(data)
    y = pd.DataFrame(b.ravel().T)

test_size=0.2
k = 100; 
degrees = range(15)
method = OLS; lambdas=[0]
#method = lasso; lambdas=[0.1]; #lambdas=[0.001, 0.01, 0.1, 0.3]
#method = ridge; lambdas=[0.1]

# Something wrong with the bias in OLS? Ridge is weird too.
def splitdata(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size, shuffle=True)
    return X_train, X_test, y_train, y_test 

y_train_size = int((1-test_size)*y.shape[0])
y_test_size = int((test_size)*y.shape[0])

y_train = np.empty((y_train_size,k))
y_test = np.empty((y_test_size,k))
y_train_pred = np.empty((y_train_size,k))
y_test_pred = np.empty((y_test_size,k))

for lamb in lambdas: # Iterate over all hyperparemeters lambda
    k_errors = np.zeros((4, 2, len(degrees))) # Initializing error array
    for d, degree in enumerate(degrees): # Iterate over polynomial complexity
        for k in range(k): # K-fold test. Do it k times and check errors.
            data_train, data_test, y_tra, y_tes = splitdata(data,y) # Sample data
            X_train = design(data_train,degree) # Design matrix for training data
            X_test = design(data_test,degree) # Design matrix for training data
            
            y_train[:,k] = y_tra[0] # Index stuff
            y_test[:,k] = y_tes[0]

            y_train_pred[:,k], y_test_pred[:,k], beta = method(X_train,y_train[:,k],X_test,y_test[:,k],lamb)

            
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
            
    filename=str(method)+str(lamb)+".png"
    error(k_errors, degrees, filename, lamb, p, True)

plt.show()
plt.savefig(str(method.__name__)+".png")
#plotstuff(y_train_pred[:,-1],y_test_pred[:,-1])


