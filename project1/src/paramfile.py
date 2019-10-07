import numpy as np
from functions import *

# Plot surface for all methods on Franke data
def params(): 
    dataset = "franke"  # Terrain or franke data
    degrees = [5] #range(15) # List of polynomial degrees
    methods = [OLS, ridge, lasso]     # Which methods to include
    lambdas = [1e-5]
    noise = 0.05        # Standard deviation of noise for franke function data
    kfolds = 5          # Number of k folds
    test = False        # Use Scikit learn methods instead
    plotterrain_ = True  # Plot terrain data
    ploterror_ = False    # Plot MSE, Bias-variance
    plotbetas_ = False   # Plot betas per lambda

    return dataset, noise, degrees, methods, lambdas, plotterrain_, ploterror_, plotbetas_, kfolds, test

# Plot surface for all methods on Terrain data
def params(): 
    dataset = "terrain"  # Terrain or franke data
    degrees = [5] #range(15) # List of polynomial degrees
    methods = [OLS, ridge, lasso]     # Which methods to include
    lambdas = [1e-5]
    noise = 0.05        # Standard deviation of noise for franke function data
    kfolds = 5          # Number of k folds
    test = False        # Use Scikit learn methods instead
    plotterrain_ = True  # Plot terrain data
    ploterror_ = False    # Plot MSE, Bias-variance
    plotbetas_ = False   # Plot betas per lambda

    return dataset, noise, degrees, methods, lambdas, plotterrain_, ploterror_, plotbetas_, kfolds, test


# Plot error over all degrees using all methods
def params(): 
    dataset = "franke"  # Terrain or franke data
    degrees = range(15) # List of polynomial degrees
    methods = [OLS, ridge, lasso]     # Which methods to include
    lambdas = [1e-5]
    noise = 0.05        # Standard deviation of noise for franke function data
    kfolds = 5          # Number of k folds
    test = False        # Use Scikit learn methods instead
    plotterrain_ = False  # Plot terrain data
    ploterror_ = True    # Plot MSE, Bias-variance
    plotbetas_ = False   # Plot betas per lambda
    return dataset, noise, degrees, methods, lambdas, plotterrain_, ploterror_, plotbetas_, kfolds, test


# For plotting betas per lambda
def params():
    dataset = "franke"  # Terrain or franke data
    degrees = [5] # List of polynomial degrees
    methods = [ridge, lasso]     # Which methods to include
    lambdas=np.logspace(-5,0,1000)
    noise = 0.05        # Standard deviation of noise for franke function data
    kfolds = 5          # Number of k folds
    test = False        # Use Scikit learn methods instead
    plotterrain_ = False  # Plot terrain data
    ploterror_ = False    # Plot MSE, Bias-variance
    plotbetas_ = True   # Plot betas per lambda
    return dataset, noise, degrees, methods, lambdas, plotterrain_, ploterror_, plotbetas_, kfolds, test

# Check OLS against sklearn
def params():
    dataset = "franke"  # Terrain or franke data
    degrees = [5] # List of polynomial degrees
    methods = [OLS]     # Which methods to include
    lambdas = [0]       # Not used here
    noise = 0.05        # Standard deviation of noise for franke function data
    kfolds = 5          # Number of k folds
    test = True        # Use Scikit learn methods instead
    plotterrain_ = False  # Plot terrain data
    ploterror_ = False    # Plot MSE, Bias-variance
    plotbetas_ = False   # Plot betas per lambda
    return dataset, noise, degrees, methods, lambdas, plotterrain_, ploterror_, plotbetas_, kfolds, test
"""
# Check Ridge against sklearn
def params():
    dataset = "franke"  # Terrain or franke data
    degrees = [5] # List of polynomial degrees
    methods = [ridge]     # Which methods to include
    lambdas = [1e-5]       # Not used here
    noise = 0.05        # Standard deviation of noise for franke function data
    kfolds = 5          # Number of k folds
    test = True        # Use Scikit learn methods instead
    plotterrain_ = False  # Plot terrain data
    ploterror_ = False    # Plot MSE, Bias-variance
    plotbetas_ = False   # Plot betas per lambda
    return dataset, noise, degrees, methods, lambdas, plotterrain_, ploterror_, plotbetas_, kfolds, test

"""
