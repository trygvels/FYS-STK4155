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

def OLS(X_train, y_train, X_test, y_test, lamb, test):
    if test==False:
        beta = svd_solver(X_train,y_train)  # Solve for x using ONLY training data
        y_train_pred = X_train.dot(beta).ravel() ## Check how beta matches training data
        y_test_pred = X_test.dot(beta).ravel() # Test (old) x on test data
        intercept = 0 
    else: # OLS by scikit learn
        fit = LinearRegression().fit(X_train, y_train)
        beta = fit.coef_
        intercept = fit.intercept_
        y_train_pred = fit.predict(X_train).reshape(y_train.shape)
        y_test_pred = fit.predict(X_test).reshape(y_test.shape)
    return y_train_pred, y_test_pred, beta,intercept
    
def ridge(X_train,y_train,X_test,y_test, lamb,test):
    if test==False: # Ridge regression from Linear algebra calculations
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

def lasso(X_train, y_train, X_test,y_test,lamb, test):
    # Lasso regression implemented using Scikit Learn
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

def rebin(a, shape): # Averege downsample
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)

def splitdata(X,y): # Initial resampling
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size, shuffle=True)
    return X_train, X_test, y_train, y_test 

def kfold(X,nsplits, testsize,trainsize, ylen): # Implementation of K-fold split.
    # Shuffle indices, chose first fracation as test data.
    train_idx = np.zeros((nsplits, trainsize))
    test_idx = np.zeros((nsplits, testsize))
    X=list(range(ylen))
    shuffle(X)
    for k in range(nsplits): # Chose test data, remove it from array.
        X_duplicate = X.copy()
        test_idx[k,:] = X_duplicate[testsize*k:testsize*(k+1)]
        del X_duplicate[testsize*k:testsize*(k+1)]
        train_idx[k,:] = X_duplicate
    return train_idx.astype(int), test_idx.astype(int)
