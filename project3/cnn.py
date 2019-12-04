"""Implementation of convolution forward and backward pass"""
import sys
import numpy as np
import functions
import numba

# def forward(conf, X_batch, params, is_training, stride = None, padding=None):
# def forward(input_layer, weight, bias, pad_size=1, stride=1, is_training=False):


def forward(conf, X_batch, params, is_training=False, pad_size=1, stride=1):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of M data points, each with C channels, height H and
    width W. We convolve each input with C_o different filters, where each filter
    spans all C_i channels and has height H_w and width W_w.

    Args:
        input_layer: The input layer with shape (batch_size, channels_x, height_x, width_x)
        weight: Filter kernels with shape (num_filters, channels_x, height_w, width_w)
        bias: Biases of shape (num_filters)
    Returns:
        output_layer: The output layer with shape (batch_size, num_filters, height_y, width_y)
    """

    n = conf["layer_dimensions"]
    L = len(n) - 1

    features = {}
    A = X_batch
    features["A_0"] = A

    A_prev = A.copy()
    weight = params["W_1"]
    bias = params["b_1"]

    # Padding width and height
    (batch_size, channels_x, height_x, width_x) = X_batch.shape  # Input shape
    input_padded = np.pad(
        X_batch,
        ((0, 0), (0, 0), (pad_size, pad_size), (pad_size, pad_size)),
        mode="constant",
    )  # Pad input
    (num_filters, channels_x, height_w, width_w) = weight.shape  # Weight shape
    height_y = 1 + (height_x + 2 * pad_size - height_w) // stride
    width_y = 1 + (width_x + 2 * pad_size - width_w) // stride
    hw_2 = height_w // 2
    ww_2 = width_w // 2

    Z = np.zeros((batch_size, num_filters, height_y, width_y))

    """
    Forward propagation through convolutional layer
    """
    for i in range(num_filters):  # Loop over filters
        for p in range(0, height_y):  # Loop over height
            for q in range(0, width_y):  # Loop over width
                Z[:, i, p, q] += np.sum(
                    input_padded[
                        :,
                        :,
                        p * stride
                        - hw_2
                        + max(pad_size, hw_2) : p * stride
                        + hw_2
                        + max(pad_size, hw_2)
                        + 1,
                        q * stride
                        - ww_2
                        + max(pad_size, ww_2) : q * stride
                        + ww_2
                        + max(pad_size, ww_2)
                        + 1,
                    ]
                    * weight[i, :, :, :],
                    axis=(1, 2, 3),
                )

        Z[:, i, :, :] += bias[np.newaxis, i, np.newaxis, np.newaxis]

    A = functions.activation(Z.copy(), "relu")
    A = A.reshape(A_prev.shape[0], -1)  # Dense fully connected layer
    if is_training:
        features["Z_1"] = Z.copy()
        features["A_1"] = A.copy()

    """
    Forward propagation through dense output layer
    """
    A_prev = A.copy()
    # print("W2.T {} and A_prev {}".format(params['W_2'].T.shape, A_prev.shape))
    Z = (
        np.dot(A_prev, params["W_2"]).T + params["b_2"]
    )  # Activation ??? This was changed to not transposed

    A = functions.softmax(Z.copy())
    if is_training:
        features["Z_2"] = Z.copy()
        features["A_2"] = A.copy()
    Y_proposed = A
    return Y_proposed, features


def backward(conf, Y_proposed, Y_reference, params, features, pad_size=1):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Args:
        output_layer_gradient: Gradient of the loss L wrt the next layer y, with shape
            (batch_size, num_filters, height_y, width_y)
        X_batch: Input layer x with shape (batch_size, channels_x, height_x, width_x)
        weight: Filter kernels with shape (num_filters, channels_x, height_w, width_w)
        bias: Biases of shape (num_filters)

    Returns:
        X_batch_gradient: Gradient of the loss L with respect to the input layer x
        weight_gradient: Gradient of the loss L with respect to the filters w
        bias_gradient: Gradient of the loss L with respect to the biases b
    """
    n_y, m = Y_reference.shape
    grad_params = {}

    dZ = Y_proposed - Y_reference  # Gradient of L (Last) softmax
    grad_params["grad_W_2"] = np.dot(features["A_1"].T, dZ.T) / m
    grad_params["grad_b_2"] = (
        np.sum(dZ, axis=1, keepdims=True) / m
    )  # Summing over one axis the same as multiplying by 1d.

    # dZ is (10,batch_size) for DNN
    output_layer_gradient = functions.activation_derivative(features["Z_1"], "relu")
    # print("gZ {}".format(gZ.shape))
    # print("W_2.T {}, dZ {}".format(params['W_2'].T.shape, dZ.shape))

    # wdZ = np.sum( (params['W_2'].T)[:,:,np.newaxis]*dZ[:,np.newaxis,:], axis=0)
    # dZ = gZ*wdZ
    # output_layer_gradient = dZ

    """
    Values in DNN
    gZ (32, 128)
    W_3.T[:,:,np.newaxis] (10, 32, 1)
    dZ[:,np.newaxis,:] (32, 1, 128)
    wdZ (32, 128)
    new dZ (32, 128)
    """

    # grad_params["grad_W_1"] = np.dot(features['A_1'], dZ.T)/m #
    # grad_params["grad_b_1"] = np.sum(dZ,axis=1,keepdims=True)/m #Summing over one axis the same as multiplying by 1d.

    batch_size, channels_y, height_y, width_y = output_layer_gradient.shape
    weight = params["W_1"]
    bias = params["b_1"]
    X_batch = features["A_0"]
    num_filters, channels_w, height_w, width_w = weight.shape

    bias_gradient = np.zeros((bias.shape))
    weight_gradient = np.zeros((weight.shape))
    X_batch_gradient = np.zeros((X_batch.shape))

    # Padding width and height
    input_padded = np.pad(
        X_batch,
        ((0, 0), (0, 0), (pad_size, pad_size), (pad_size, pad_size)),
        mode="constant",
    )
    output_gradient_padded = np.pad(
        output_layer_gradient,
        ((0, 0), (0, 0), (pad_size, pad_size), (pad_size, pad_size)),
        mode="constant",
    )

    hw_2 = height_w // 2
    ww_2 = width_w // 2

    # Update weights and biases with backprop
    weight_gradient, bias_gradient, X_batch_gradient = hehe(
        weight,
        batch_size,
        channels_y,
        height_y,
        width_y,
        output_layer_gradient,
        bias_gradient,
        channels_w,
        height_w,
        width_w,
        weight_gradient,
        input_padded,
        X_batch_gradient,
        output_gradient_padded,
    )
    """
    This is the convolutional layer
    """

    grad_params["grad_b_1"] = bias_gradient.copy()
    grad_params["grad_W_1"] = weight_gradient.copy()

    return grad_params


@numba.njit(cache=True)
def hehe(
    weight,
    batch_size,
    channels_y,
    height_y,
    width_y,
    output_layer_gradient,
    bias_gradient,
    channels_w,
    height_w,
    width_w,
    weight_gradient,
    input_padded,
    X_batch_gradient,
    output_gradient_padded,
):
    weight2 = weight[:, :, ::-1, ::-1].copy()  # Flip filter
    for i in range(batch_size):  # Loop over every batch
        for j in range(channels_y):  # Loop over each color channel
            for p in range(height_y):  # Loop over height
                for q in range(width_y):  # Loop over width
                    bias_gradient[j] += output_layer_gradient[i, j, p, q]
                    for k in range(channels_w):  # Loop over filter channe
                        for r in range(height_w):  # Loop over filter height
                            for s in range(width_w):  # Loop over filter width
                                weight_gradient[j, k, r, s] += (
                                    output_layer_gradient[i, j, p, q]
                                    * input_padded[i, k, p + r, q + s]
                                )
                                X_batch_gradient[i, k, p, q] += (
                                    output_gradient_padded[i, j, p + r, q + s]
                                    * weight2[j, k, r, s]
                                )
    return weight_gradient, bias_gradient, X_batch_gradient


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