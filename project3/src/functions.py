"""Define the dense neural network model"""
import numpy as np
from scipy.stats import truncnorm
import sys

np.random.seed(42069)


def one_hot(Y, num_classes):

    m = len(Y)
    Y_tilde = np.zeros((num_classes, m))
    Y_tilde[Y, np.arange(m)] = 1
    return Y_tilde


def initialization(conf):
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

        channels_x = conf["channels_x"]
        height_x = conf["height_x"]
        width_x = conf["width_x"]
        height_w = conf["height_w"]
        width_w = conf["width_w"]
        num_filters = conf["num_filters"]
        stride = conf["stride"]
        pad_size = conf["pad_size"]
        height_y = 1 + (height_x + 2 * pad_size - height_w) // stride
        width_y = 1 + (width_x + 2 * pad_size - width_w) // stride
        n[0] = num_filters * height_y * width_y  # DNN input from CNN

        params_cnn["W_1"] = np.random.normal(
            mu, np.sqrt(2.0 / (channels_x * height_w * width_w)), (num_filters, channels_x, height_w, width_w),
        )
        params_cnn["b_1"] = np.zeros((num_filters,))

        n = conf["layer_dimensions"]
        for l in range(1, len(n)):
            sigma2 = 2.0 / n[l - 1]

            params_dnn["W_" + str(l)] = np.random.normal(mu, np.sqrt(sigma2), (n[l - 1], n[l]))
            params_dnn["b_" + str(l)] = np.zeros((n[l], 1))
    return params_dnn, params_cnn


def activation(Z, activation_function):
    if activation_function == "relu":
        Z[np.where(Z < 0)] = 0.0
        return Z
    else:
        print("Error: Unimplemented activation function: {}", activation_function)
        return None


def softmax(Z):
    ZRED = Z - np.max(Z)
    ZEXP = np.exp(ZRED)

    ZSUM = ZEXP.sum(axis=0)
    logZ = np.log(ZEXP) - np.log(ZSUM)
    return np.exp(logZ)


def activation_derivative(Z, activation_function):

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
    n_y, m = Y_reference.shape
    cost = -np.sum((Y_reference * np.log(Y_proposed))) / m
    refcorr = np.argmax(Y_reference, axis=0)
    propcorr = np.argmax(Y_proposed, axis=0)
    num_correct = sum(refcorr == propcorr)

    return cost, num_correct


def optimize(conf, params, grad_params, adams):
    if conf["optimizer"] == "adam":
        params, conf, adams = adam(conf, params, grad_params, adams)
    else:
        params, conf, adams = gradient_descent_update(conf, params, grad_params)
    return params, conf, adams


def gradient_descent_update(conf, params, grad_params):
    learning_rate = conf["learning_rate"]
    adams = {}  # Not used
    updated_params = {}
    for key in params:
        updated_params[key] = params[key] - learning_rate * grad_params["grad_" + key]

    return updated_params, conf, adams


def adam(conf, params, grad_params, adams):
    beta1 = 0.9
    beta2 = 0.999
    alpha = 0.001
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
