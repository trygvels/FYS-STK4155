"""Define the dense neural network model"""
import numpy as np
from scipy.stats import truncnorm
import sys

np.random.seed(42069)


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
    """
    Initializes the weights and biases for the network. 
    We use ReLU in this project, so weights are initialized using He initialization
    """
    params_dnn = {}
    params_cnn = {}
    mu = 0.0
    if conf["net"] == "DNN":
        n = conf["layer_dimensions"]
        for l in range(1, len(n)):
            sigma2 = 2.0 / n[l - 1]

            params_dnn["W_" + str(l)] = np.random.normal(mu, np.sqrt(sigma2), (n[l - 1], n[l]))
            params_dnn["b_" + str(l)] = np.zeros((n[l], 1))

    else:
        n = conf["layer_dimensions"]

        # Get size parameters
        channels_x = conf["channels_x"]
        height_x = conf["height_x"]
        width_x = conf["width_x"]
        height_w = conf["height_w"]
        width_w = conf["width_w"]
        num_filters = conf["num_filters"]
        stride = conf["stride"]
        pad_size = conf["pad_size"]

        # Calculate size of feature layers given pad and stride
        height_y = 1 + (height_x + 2 * pad_size - height_w) // stride
        width_y = 1 + (width_x + 2 * pad_size - width_w) // stride
        n[0] = num_filters * height_y * width_y  # DNN input from CNN

        # Initialize weights of conv layer
        params_cnn["W_1"] = np.random.normal(
            mu, np.sqrt(2.0 / (channels_x * height_w * width_w)), (num_filters, channels_x, height_w, width_w),
        )
        params_cnn["b_1"] = np.zeros((num_filters,))

        # Initialize weights of fully connected part
        n = conf["layer_dimensions"]
        for l in range(1, len(n)):
            sigma2 = 2.0 / n[l - 1]

            params_dnn["W_" + str(l)] = np.random.normal(mu, np.sqrt(sigma2), (n[l - 1], n[l]))
            params_dnn["b_" + str(l)] = np.zeros((n[l], 1))
    return params_dnn, params_cnn


def activation(Z, activation_function):
    """
    Activation functions.
    Only relu is used for this project.
    """
    if activation_function == "relu":
        Z[np.where(Z < 0)] = 0.0
        return Z
    else:
        print("Error: Unimplemented activation function: {}", activation_function)
        return None


def softmax(Z):
    """
    Softmax function for converting outputs to probabilities.

    To improve numerical stability, we do the following

    1: Subtract Z from max(Z) in the exponentials
    2: Take the logarithm of the whole softmax, and then take the exponential of that in the end

    """

    ZRED = Z - np.max(Z)
    ZEXP = np.exp(ZRED)

    ZSUM = ZEXP.sum(axis=0)
    logZ = np.log(ZEXP) - np.log(ZSUM)
    return np.exp(logZ)


def activation_derivative(Z, activation_function):
    """
    Derivatives of activation functions for backprop.
    Only ReLU is used in this project.
    """
    if activation_function == "relu":
        Z[np.where(Z >= 0)] = 1.0
        Z[np.where(Z < 0)] = 0.0
        return Z
    else:
        print(
            "Error: Unimplemented derivative of activation function: {}", activation_function,
        )
        return None


def cross_entropy_cost(Y_proposed, Y_reference):
    """
    Computes the cross entropy cost.
    Y is on the form (classes, batch_size)
    """
    # (classes, batch size)
    n_y, m = Y_reference.shape
    cost = -np.sum((Y_reference * np.log(Y_proposed))) / m
    refcorr = np.argmax(Y_reference, axis=0)
    propcorr = np.argmax(Y_proposed, axis=0)
    num_correct = sum(refcorr == propcorr)

    return cost, num_correct


def optimize(conf, params, grad_params, adams):
    """
    Runs the chosen optimizer.
    Gradient descent or adam.
    """
    if conf["optimizer"] == "adam":
        params, conf, adams = adam(conf, params, grad_params, adams)
    else:
        params, conf, adams = gradient_descent_update(conf, params, grad_params)
    return params, conf, adams


def gradient_descent_update(conf, params, grad_params):
    """
    Gradient descent update which takes the specified learning rate from configuration.
    """
    learning_rate = conf["learning_rate"]
    adams = {}  # Not used
    updated_params = {}
    for key in params:
        updated_params[key] = params[key] - learning_rate * grad_params["grad_" + key]

    return updated_params, conf, adams


def adam(conf, params, grad_params, adams):
    """
    An adam optimizer.
    Drastically improves training time.
    Uses different learning weights for each parameter and updates throughout.
    Set learning rate is not used.
    """
    alpha = conf['learning_rate']
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    updated_params = {}
    for key in params:
        if "t_" + key not in adams:
            adams["t_" + key] = 0.0
            adams["m_" + key] = 0.0
            adams["v_" + key] = 0.0

        adams["t_" + key] = adams["t_" + key] + 1
        adams["m_" + key] = beta1 * adams["m_" + key] + (1 - beta1) * grad_params["grad_" + key]
        adams["v_" + key] = beta2 * adams["v_" + key] + (1 - beta2) * (grad_params["grad_" + key] ** 2)
        m_hat = adams["m_" + key] / (1 - beta1 ** adams["t_" + key])
        v_hat = adams["v_" + key] / (1 - beta2 ** adams["t_" + key])
        updated_params[key] = params[key] - alpha * (m_hat / (np.sqrt(v_hat) - epsilon))

    return updated_params, conf, adams
