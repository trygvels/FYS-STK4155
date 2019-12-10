import numpy as np
import sys
from functions import activation, activation_derivative, softmax


def forward(conf, X_batch, params, is_training):
    """
    Forward propagation through fully connected network.

    X_batch:
        (batch_size, channels * height * width)
    """
    n = conf["layer_dimensions"]
    L = len(n) - 1

    # Saves the input
    A = X_batch
    features = {}
    features["A_0"] = A

    # Loop over each layer in network
    for l in range(1, L + 1):
        A_prev = A.copy()
        Z = np.dot(params["W_" + str(l)].T, A_prev) + params["b_" + str(l)]

        # Calculates activation (Relu, or softmax for output)
        if l < L:
            A = activation(Z.copy(), "relu")
        else:
            A = softmax(Z.copy())
        if is_training:
            # Save activations if training
            features["Z_" + str(l)] = Z.copy()
            features["A_" + str(l)] = A.copy()

    # Y_proposed is the probabilities returned by passing
    # activations through the softmax function.
    Y_proposed = A
    return Y_proposed, features


def backward(conf, Y_proposed, Y_reference, params, features):
    """
    Backpropagation through the fully connected network.
    """
    n_y, m = Y_reference.shape
    n = conf["layer_dimensions"]
    L = len(n) - 1
    grad_params = {}

    # Output layer gradient. Gradient of loss using softmax.
    dZ = Y_proposed - Y_reference

    # Loop backwards through layers
    for l in reversed(range(1, L + 1)):
        # Gradient of weight l
        grad_params["grad_W_" + str(l)] = np.dot(features["A_" + str(l - 1)], dZ.T) / m  #
        # Gradient of bias l
        grad_params["grad_b_" + str(l)] = np.sum(dZ, axis=1, keepdims=True) / m

        # Calculate new output gradient
        if l > 1:
            gZ = activation_derivative(features["Z_" + str(l - 1)], "relu")
            wdZ = np.sum((params["W_" + str(l)].T)[:, :, np.newaxis] * dZ[:, np.newaxis, :], axis=0)
            dZ = gZ * wdZ
    return grad_params, dZ
