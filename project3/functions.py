"""Define the dense neural network model"""
import numpy as np
from scipy.stats import truncnorm


def one_hot(Y, num_classes):
    """Perform one-hot encoding on input Y.

    It is assumed that Y is a 1D numpy array of length m_b (batch_size) with integer values in
    range [0, num_classes-1]. The encoded matrix Y_tilde will be a [num_classes, m_b] shaped matrix
    with values

                   | 1,  if Y[i] = j
    Y_tilde[i,j] = |
                   | 0,  else
    """
    m = len(Y)
    Y_tilde = np.zeros((num_classes, m))
    Y_tilde[Y, np.arange(m)] = 1
    return Y_tilde


def initialization(conf):
    """Initialize the parameters of the network.

    Args:
        layer_dimensions: A list of length L+1 with the number of nodes in each layer, including
                          the input layer, all hidden layers, and the output layer.
    Returns:
        params: A dictionary with initialized parameters for all parameters (weights and biases) in
                the network.
    """
    params = {}
    mu = 0.0
    if conf["network_type"] == "DNN":
        n = conf["layer_dimensions"]
        for l in range(1, len(n)):
            sigma2 = 2.0 / n[l - 1]

            params["W_" + str(l)] = np.random.normal(
                mu, np.sqrt(sigma2), (n[l - 1], n[l])
            )
            params["b_" + str(l)] = np.zeros((n[l], 1))
    else:
        n = conf["layer_dimensions"]
        batch_size = conf["batch_size"]
        num_filters = conf["num_filters"]
        channels_x = conf["channels_x"]
        height_x = conf["height_x"]
        width_x = conf["width_x"]
        height_w = conf["height_w"]
        width_w = conf["width_w"]

        # Init weights for convolutional layer

        params["W_1"] = np.random.normal(
            mu,
            np.sqrt(2.0 / (channels_x * height_w * width_w)),
            (num_filters, channels_x, height_w, width_w),
        )
        params["b_1"] = np.zeros((num_filters,))

        # Init weights for dense output layer
        hehe = height_x * width_x * num_filters
        params["W_2"] = np.random.normal(mu, np.sqrt(2.0 / hehe), (hehe, n[-1]))
        params["b_2"] = np.zeros((n[-1], 1))

    return params


def activation(Z, activation_function):
    """Compute a non-linear activation function.

    Args:
        Z: numpy array of floats with shape [n, m]
    Returns:
        numpy array of floats with shape [n, m]
    """
    if activation_function == "relu":
        Z[np.where(Z < 0)] = 0.0
        return Z
    else:
        print("Error: Unimplemented activation function: {}", activation_function)
        return None


def softmax(Z):
    """Compute and return the softmax of the input.

    To improve numerical stability, we do the following

    1: Subtract Z from max(Z) in the exponentials
    2: Take the logarithm of the whole softmax, and then take the exponential of that in the end

    Args:
        Z: numpy array of floats with shape [n, m]
    Returns:
        numpy array of floats with shape [n, m]
    """
    ZRED = Z - np.max(Z)
    ZEXP = np.exp(ZRED)
    ZSUM = ZEXP.sum(axis=0)
    logZ = np.log(ZEXP) - np.log(ZSUM)
    return np.exp(logZ)


def activation_derivative(Z, activation_function):
    """Compute the gradient of the activation function.

    Args:
        Z: numpy array of floats with shape [n, m]
    Returns:
        numpy array of floats with shape [n, m]
    """
    if activation_function == "relu":
        Z[np.where(Z >= 0)] = 1.0
        Z[np.where(Z < 0)] = 0.0
        return Z
    else:
        print(
            "Error: Unimplemented derivative of activation function: {}",
            activation_function,
        )
        return None


def cross_entropy_cost(Y_proposed, Y_reference):
    """Compute the cross entropy cost function.

    Args:
        Y_proposed: numpy array of floats with shape [n_y, m].
        Y_reference: numpy array of floats with shape [n_y, m]. Collection of one-hot encoded
                     true input labels

    Returns:
        cost: Scalar float: 1/m * sum_i^m sum_j^n y_reference_ij log y_proposed_ij
        num_correct: Scalar integer
    """
    n_y, m = Y_reference.shape
    cost = -np.sum((Y_reference * np.log(Y_proposed))) / m
    refcorr = np.argmax(Y_reference, axis=0)
    propcorr = np.argmax(Y_proposed, axis=0)
    # print(Y_proposed)
    # print(Y_reference)
    num_correct = sum(refcorr == propcorr)
    return cost, num_correct
