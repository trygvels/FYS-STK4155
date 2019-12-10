import numpy as np
import sys
from functions import activation, activation_derivative, softmax


def forward(conf, X_batch, params, is_training):
    n = conf["layer_dimensions"]
    L = len(n) - 1

    A = X_batch
    features = {}
    features["A_0"] = A
    for l in range(1, L + 1):  # For each layer in network (4 total)
        A_prev = A.copy()
        Z = np.dot(params["W_" + str(l)].T, A_prev) + params["b_" + str(l)]  # Activation

        if l < L:
            A = activation(Z.copy(), "relu")  # Output
        else:
            A = softmax(Z.copy())
        if is_training:
            features["Z_" + str(l)] = Z.copy()
            features["A_" + str(l)] = A.copy()

    Y_proposed = A
    return Y_proposed, features


def backward(conf, Y_proposed, Y_reference, params, features):
    n_y, m = Y_reference.shape
    n = conf["layer_dimensions"]
    L = len(n) - 1
    grad_params = {}

    dZ = Y_proposed - Y_reference  # Gradient of L (Last) softmax

    for l in reversed(range(1, L + 1)):
        grad_params["grad_W_" + str(l)] = np.dot(features["A_" + str(l - 1)], dZ.T) / m  #
        grad_params["grad_b_" + str(l)] = np.sum(dZ, axis=1, keepdims=True) / m
        if l > 1:
            gZ = activation_derivative(features["Z_" + str(l - 1)], "relu")
            wdZ = np.sum((params["W_" + str(l)].T)[:, :, np.newaxis] * dZ[:, np.newaxis, :], axis=0)
            dZ = gZ * wdZ
    return grad_params, dZ
