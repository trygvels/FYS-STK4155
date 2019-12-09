import sys
import numpy as np
import functions
import numba

np.random.seed(42069)


def forward(conf, input_layer, params, is_training=False):
    weight = params["W_1"]
    bias = params["b_1"]
    stride = conf["stride"]
    pad_size = conf["pad_size"]

    # Padding width and height
    (batch_size, channels_x, height_x, width_x) = input_layer.shape
    input_padded = np.pad(input_layer, ((0, 0), (0, 0), (pad_size, pad_size), (pad_size, pad_size)), mode="constant")
    (num_filters, channels_w, height_w, width_w) = weight.shape

    height_y = 1 + (height_x + 2 * pad_size - height_w) // stride
    width_y = 1 + (width_x + 2 * pad_size - width_w) // stride
    output_layer = np.zeros((batch_size, num_filters, height_y, width_y))

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

    Z = output_layer.copy()
    A = functions.activation(Z.copy(), "relu")

    if is_training:
        features["Z_1"] = Z.copy()
        features["A_1"] = A.copy()

    return A, features


def backward(dZ, input_layer, params, params2, conf, features):
    input_layer = features["A_0"]
    weight = params["W_1"]
    bias = params["b_1"]
    stride = conf["stride"]
    pad_size = conf["pad_size"]

    # Compute output layer gradient
    gZ = functions.activation_derivative(features["Z_1"], "relu")
    wdZ = np.sum((params2["W_1"])[np.newaxis, :, :] * dZ.T[:, np.newaxis, :], axis=2,)
    dZ = gZ * wdZ.reshape(gZ.shape)
    output_layer_gradient = dZ

    batch_size, channels_y, height_y, width_y = output_layer_gradient.shape
    batch_size, channels_x, height_x, width_x = input_layer.shape
    num_filters, channels_w, height_w, width_w = weight.shape

    bias_gradient = np.zeros((bias.shape))
    weight_gradient = np.zeros((weight.shape))
    input_layer_gradient = np.zeros((input_layer.shape))

    # Padding width and height
    input_padded = np.pad(input_layer, ((0, 0), (0, 0), (pad_size, pad_size), (pad_size, pad_size)), mode="constant")
    output_gradient_padded = np.pad(
        output_layer_gradient, ((0, 0), (0, 0), (pad_size, pad_size), (pad_size, pad_size)), mode="constant"
    )

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
    # input_layer_gradient,
    grad_params["grad_W_1"] = weight_gradient / batch_size
    grad_params["grad_b_1"] = bias_gradient / batch_size

    return grad_params


"""
Forward pass loop written in for loop format for numba
"""


@numba.njit(cache=True)
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
    # Numpa friendly forward pass
    for b in range(batch_size):
        for n in range(num_filters):
            output_layer[b, n] = bias[n]
            for c in range(channels_x):
                for wy in range(width_y):
                    for hy in range(height_y):
                        temp = 0
                        for hw in range(width_w):
                            for ww in range(height_w):
                                temp += input_padded[b, c, hy * stride + hw, wy * stride + ww] * weight[n, c, hw, ww]
                        output_layer[b, n, hy, wy] += temp
    return output_layer


"""
Backprop loop written in numba friendly format
"""


@numba.njit(cache=True)
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

                                # Dont need to calculate input layer gradient when not going farther
                                # input_layer_gradient[i, k, p, q] += (
                                #    output_gradient_padded[i, j, p + r, q + s] * weight2[j, k, r, s]
                                # )

    return weight_gradient, bias_gradient, input_layer_gradient
