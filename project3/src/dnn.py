import numpy as np
import sys
from functions import activation, activation_derivative, softmax


def forward(conf, X_batch, params, is_training):
    """One forward step.

    Args:
        conf: Configuration dictionary.
        X_batch: float numpy array with shape [n^[0], batch_size]. Input image batch.
        params: python dict with weight and bias parameters for each layer.
        is_training: Boolean to indicate if we are training or not. This function can namely be
                     used for inference only, in which case we do not need to store the features
                     values.

    Returns:
        Y_proposed: float numpy array with shape [n^[L], batch_size]. The output predictions of the
                    network, where n^[L] is the number of prediction classes. For each input i in
                    the batch, Y_proposed[c, i] gives the probability that input i belongs to class
                    c.
        features: Dictionary with
                - the linear combinations Z^[l] = W^[l]a^[l-1] + b^[l] for l in [1, L].
                - the activations A^[l] = activation(Z^[l]) for l in [1, L].
               We cache them in order to use them when computing gradients in the backpropagation.
    """

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
    """Update parameters using backpropagation algorithm.

    Args:
        conf: Configuration dictionary.
        Y_proposed: numpy array of floats with shape [n_y, m].
        features: Dictionary with matrices from the forward propagation. Contains
                - the linear combinations Z^[l] = W^[l]a^[l-1] + b^[l] for l in [1, L].
                - the activations A^[l] = activation(Z^[l]) for l in [1, L].
        params: Dictionary with values of the trainable parameters.
                - the weights W^[l] for l in [1, L].
                - the biases b^[l] for l in [1, L].
    Returns:
        grad_params: Dictionary with matrices that is to be used in the parameter update. Contains
                - the gradient of the weights, grad_W^[l] for l in [1, L].
                - the gradient of the biases grad_b^[l] for l in [1, L].
    """

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


def gradient_descent_update(conf, params, grad_params):
    """Update the parameters in params according to the gradient descent update routine.

    Args:
        conf: Configuration dictionary
        params: Parameter dictionary with W and b for all layers
        grad_params: Parameter dictionary with b gradients, and W gradients for all
                     layers.
    Returns:
        params: Updated parameter dictionary.
    """
    learning_rate = conf["learning_rate"]

    updated_params = {}
    for key in params:
        updated_params[key] = params[key] - learning_rate * grad_params["grad_" + key]

    return updated_params
