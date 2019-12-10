import sys
import numpy as np
import functions
import numba

np.random.seed(42069)


def forward(conf, input_layer, params, is_training=False):
    """
    Forward propagation through the Convolutional layer.

    input_layer:
        (batch_size, channels_x, height_x, width_x)
    """

    # Get weights an parameters
    weight = params["W_1"]
    bias = params["b_1"]
    # Get parameters
    stride = conf["stride"]
    pad_size = conf["pad_size"]

    # Padding width and height
    (batch_size, channels_x, height_x, width_x) = input_layer.shape
    input_padded = np.pad(input_layer, ((0,), (0,), (pad_size,), (pad_size,)), mode="constant")
    (num_filters, channels_w, height_w, width_w) = weight.shape

    # Calculate dimensions of output layer and initialize
    height_y = 1 + (height_x + 2 * pad_size - height_w) // stride
    width_y = 1 + (width_x + 2 * pad_size - width_w) // stride
    output_layer = np.zeros((batch_size, num_filters, height_y, width_y))

    # Save input layer
    A = input_layer
    features = {}
    features["A_0"] = A

    # Forward pass loop in numba
    output_layer = forwardloop(
        batch_size,
        num_filters,
        output_layer,
        weight,
        bias,
        input_padded,
        channels_x,
        width_y,
        height_y,
        width_w,
        height_w,
        stride,
    )

    # Save output to Z
    Z = output_layer.copy()
    A = functions.activation(Z.copy(), "relu")

    if is_training:
        # If training, save outputs
        features["Z_1"] = Z.copy()
        features["A_1"] = A.copy()

    return A, features


def backward(dZ, params, params2, conf, features):
    """
    Backward propagation through cnn layer.

    """
    # Get weights, biases and original image
    input_layer = features["A_0"]
    weight = params["W_1"]
    bias = params["b_1"]
    # Get parameters
    stride = conf["stride"]
    pad_size = conf["pad_size"]

    # Compute output layer gradient
    gZ = functions.activation_derivative(features["Z_1"], "relu")
    wdZ = np.sum((params2["W_1"])[np.newaxis, :, :] * dZ.T[:, np.newaxis, :], axis=2,)
    dZ = gZ * wdZ.reshape(gZ.shape)
    output_layer_gradient = dZ

    # Get dimensions
    batch_size, channels_y, height_y, width_y = output_layer_gradient.shape
    num_filters, channels_w, height_w, width_w = weight.shape

    # Initialize gradient arrays
    bias_gradient = np.zeros((bias.shape))
    weight_gradient = np.zeros((weight.shape))
    input_layer_gradient = np.zeros((input_layer.shape))

    # Padding width and height
    input_padded = np.pad(input_layer, ((0,), (0,), (pad_size,), (pad_size,)), mode="constant")
    output_gradient_padded = np.pad(output_layer_gradient, ((0,), (0,), (pad_size,), (pad_size,)), mode="constant")

    # Backprop loop in numba
    weight_gradient, bias_gradient, input_layer_gradient = backproploop(
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
        input_layer_gradient,
        output_gradient_padded,
    )

    grad_params = {}
    # Calculate gradient of weights
    grad_params["grad_W_1"] = weight_gradient / batch_size
    grad_params["grad_b_1"] = bias_gradient / batch_size

    return grad_params


@numba.njit(cache=True, fastmath=True)
def forwardloop(
    batch_size,
    num_filters,
    output_layer,
    weight,
    bias,
    input_padded,
    channels_x,
    width_y,
    height_y,
    width_w,
    height_w,
    stride,
):
    """
    Forward propagation loop written in numba friendly format
    """
    # Numpa friendly forward pass
    for i in range(batch_size):
        for k in range(num_filters):
            output_layer[i, k] = bias[k]
            for j in range(channels_x):
                for p in range(height_y):
                    for q in range(width_y):
                        temp = 0
                        for r in range(height_w):
                            for s in range(width_w):
                                temp += input_padded[i, j, p * stride + s, q * stride + r] * weight[k, j, s, r]
                        output_layer[i, k, p, q] += temp
    return output_layer


@numba.njit(cache=True, fastmath=True)
def backproploop(
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
    input_layer_gradient,
    output_gradient_padded,
):
    """
    Backprop loop written in numba friendly format
    """
    # Written in numba-friendly format.
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
                                    output_layer_gradient[i, j, p, q] * input_padded[i, k, p + r, q + s]
                                )
                                """
                                # Dont need to calculate input layer gradient when not going farther
                                input_layer_gradient[i, k, p, q] += (
                                        output_gradient_padded[i, j, p + r, q + s] * weight2[j, k, r, s]
                                )
                                """
    return weight_gradient, bias_gradient, input_layer_gradient
