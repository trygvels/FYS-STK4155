import os
import sys
import numpy as np

import import_data
import run

import matplotlib.pyplot as plt
import matplotlib

np.random.seed(42069)


def config():
    """Return a dict of configuration settings used in the program"""
    conf = {}

    # Determine what dataset to run on. 'mnist', 'cifar10' and 'svhn' are currently supported.
    conf["dataset"] = "cifar10"  #'cifar10' #'mnist'
    # Relevant datasets will be put in the location data_root_dir/dataset.
    conf["data_root_dir"] = "/tmp/data"
    # Type of neural net. (CNN adds one conv layer in front)
    conf["net"] = "CNN"

    # Run keras test (Runs Keras only)
    conf["keras"] = True
    conf["keras_optimal"] = False  # Uses max pooling and more convolution
    conf["epochs"] = 10  # Only used in keras

    # Number of input nodes. This is determined by the dataset in runtime.
    conf["input_dimension"] = None
    # Number of hidden layers, with the number of nodes in each layer.
    conf["hidden_dimensions"] = [128, 64]
    # Number of classes. This is determined by the dataset in runtime.
    conf["output_dimension"] = None
    # This will be determined in runtime when input_dimension and output_dimension is set.
    conf["layer_dimensions"] = None

    # Size of development partition of the training set
    conf["devel_size"] = 5000
    # What activation function to use in the nodes in the hidden layers.
    conf["activation_function"] = "relu"
    # The number of steps to run before termination of training. One step is one forward->backward
    # pass of a mini-batch
    conf["max_steps"] = 10000  # 20000
    # The batch size used in training.
    conf["batch_size"] = 128
    # The step size used by the optimization routine.
    conf["learning_rate"] = 1.0e-3

    # Whether or not to write certain things to stdout.
    conf["verbose"] = True
    # How often (in steps) to log the training progress. Prints to stdout if verbose = True.
    conf["train_progress"] = 10
    # How often (in steps) to evaluate the method on the development partition. Prints to stdout
    # if verbose = True.
    conf["devel_progress"] = 100

    # Wether or not to save predictions and figure
    conf["output"] = True
    conf["savefig"] = True
    """
    CNN PARAMETERS
    """
    if conf["net"] == "CNN":
        conf["num_filters"] = 3 #32
        conf["height_w"] = 3
        conf["width_w"] = 3
        conf["stride"] = 1
        conf["pad_size"] = 1
    else:
        conf["stride"] = None
        conf["pad_size"] = None
        # If we are not using a conv layer, add an additional dense
        conf["hidden_dimensions"] = [128, 128, 64]

    # Which optimizer to use "adam" or "GD"
    conf["optimizer"] = "adam"

    # Output filename for plot and predictions
    conf["output_filename"] = "{}_{}_{}_n{}".format(conf["dataset"], conf["net"], conf["optimizer"], conf["max_steps"])

    # Append conv layer specifications
    if conf["net"] == "CNN":
        conf["output_filename"] += "_D{}x{}_C{}x{}x{}".format(
            conf["hidden_dimensions"][0],
            conf["hidden_dimensions"][1],
            conf["num_filters"],
            conf["height_w"],
            conf["width_w"],
        )
    else:
        conf["output_filename"] += "_D{}x{}x{}".format(
            conf["hidden_dimensions"][0], conf["hidden_dimensions"][1], conf["hidden_dimensions"][2]
        )

    conf["output_filename"] += "_KERAS" if conf["keras"] else ""
    conf["output_filename"] += "-optimal" if conf["keras_optimal"] else ""

    return conf


def plot_progress(conf, train_progress, devel_progress):
    """Plot a chart of the training progress"""

    # Change font size
    font = {"size": 12}
    matplotlib.rc("font", **font)

    # Plot accuracy
    fig, ax1 = plt.subplots(figsize=(8, 6), dpi=100)
    ax1.plot(
        train_progress["steps"], train_progress["ccr"], color="C0", label="Training set accuracy",
    )
    ax1.plot(
        devel_progress["steps"], devel_progress["ccr"], color="C1", label="Development set accuracy",
    )

    xlab = "Epochs" if conf["keras"] else "Steps"
    ax1.set_xlabel(xlab)

    ax1.set_ylabel("Accuracy")

    # Plot cost on right y-axis
    ax2 = ax1.twinx()
    ax2.plot(
        train_progress["steps"], train_progress["cost"], color="C2", label="Training set cost",
    )
    ax2.set_ylabel("Cross entropy cost")

    # Grid lines for each sub plot
    ax1.yaxis.grid(linestyle=":")
    ax2.yaxis.grid(linestyle="--")

    # Legend
    ax1.legend(loc="lower left", bbox_to_anchor=(0.5, 0.52), framealpha=0.0)
    ax2.legend(loc="lower left", bbox_to_anchor=(0.5, 0.45), framealpha=0.0)

    fig.tight_layout()

    # Save figure
    out_filename = conf["output_filename"]
    if conf["savefig"]:
        plt.savefig("../figs/" + out_filename + ".png")

    # plt.show()


def get_data(conf):
    """
    Function for loading data from import_data.py and specifying correct dimensions.
    This function returns training, development and test data.
    """
    data_dir = os.path.join(conf["data_root_dir"], conf["dataset"])
    print(data_dir)
    if conf["dataset"] == "cifar10":
        conf["channels_x"] = 3
        conf["height_x"] = 32
        conf["width_x"] = 32
        conf["input_dimension"] = conf["channels_x"] * conf["height_x"] * conf["width_x"]
        conf["output_dimension"] = 10
        X_train, Y_train, X_devel, Y_devel, X_test, Y_test = import_data.load_cifar10(
            conf, data_dir, conf["devel_size"]
        )
    elif conf["dataset"] == "mnist":
        conf["channels_x"] = 1
        conf["height_x"] = 28
        conf["width_x"] = 28
        conf["input_dimension"] = conf["channels_x"] * conf["height_x"] * conf["width_x"]
        conf["output_dimension"] = 10
        X_train, Y_train, X_devel, Y_devel, X_test, Y_test = import_data.load_mnist(conf, data_dir, conf["devel_size"])
    elif conf["dataset"] == "svhn":
        conf["channels_x"] = 3
        conf["height_x"] = 32
        conf["width_x"] = 32
        conf["input_dimension"] = conf["channels_x"] * conf["height_x"] * conf["width_x"]
        conf["output_dimension"] = 10
        X_train, Y_train, X_devel, Y_devel, X_test, Y_test = import_data.load_svhn(conf, data_dir, conf["devel_size"])

    conf["layer_dimensions"] = [conf["input_dimension"]] + conf["hidden_dimensions"] + [conf["output_dimension"]]

    if conf["verbose"]:
        print("Train dataset:")
        print(
            "  shape = {}, data type = {}, min val = {}, max val = {}".format(
                X_train.shape, X_train.dtype, np.min(X_train), np.max(X_train)
            )
        )
        print("Development dataset:")
        print(
            "  shape = {}, data type = {}, min val = {}, max val = {}".format(
                X_devel.shape, X_devel.dtype, np.min(X_devel), np.max(X_devel)
            )
        )
        print("Test dataset:")
        print(
            "  shape = {}, data type = {}, min val = {}, max val = {}".format(
                X_test.shape, X_test.dtype, np.min(X_test), np.max(X_test)
            )
        )

    return X_train, Y_train, X_devel, Y_devel, X_test, Y_test


def main():
    """
    Main function for running the whole network based on the parameters set in the "conf" dictionary in the config() function.
    First, the network is trained, and checked against a development set at the prefered increments, then the training progress is plotted, before the final evaluation is done on all three data sets.
    """
    # Get parameters from config() function
    conf = config()
    # Get all data and split it into three sets. Format: (datasize, channels, height, width).
    X_train, Y_train, X_devel, Y_devel, X_test, Y_test = get_data(conf)

    # Test with keras
    if conf["keras"] == True:
        train_progress, devel_progress = run.kerasnet(conf, X_train, Y_train, X_devel, Y_devel, X_test, Y_test)
        plot_progress(conf, train_progress, devel_progress)
        sys.exit()

    # Run training and save weights and biases in params_dnn and params_cnn.
    conf, params_dnn, params_cnn, train_progress, devel_progress = run.train(conf, X_train, Y_train, X_devel, Y_devel,)

    # Plot the progress of the network over training steps
    plot_progress(conf, train_progress, devel_progress)

    # Evaluate the network on all three data sets. If output=True, then the predictions made on the test set is saved.
    print("Evaluating train set")
    num_correct, num_evaluated = run.evaluate(conf, params_dnn, params_cnn, X_train, Y_train)
    print("CCR = {0:>5} / {1:>5} = {2:>6.4f}".format(num_correct, num_evaluated, num_correct / num_evaluated))
    print("Evaluating development set")
    num_correct, num_evaluated = run.evaluate(conf, params_dnn, params_cnn, X_devel, Y_devel)
    print("CCR = {0:>5} / {1:>5} = {2:>6.4f}".format(num_correct, num_evaluated, num_correct / num_evaluated))
    print("Evaluating test set")
    num_correct, num_evaluated = run.evaluate(conf, params_dnn, params_cnn, X_test, Y_test, output=conf["output"])
    print("CCR = {0:>5} / {1:>5} = {2:>6.4f}".format(num_correct, num_evaluated, num_correct / num_evaluated))


if __name__ == "__main__":
    main()
