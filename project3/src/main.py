import os
import numpy as np

import import_data
import run

import matplotlib.pyplot as plt

np.random.seed(42069)


def config():
    """Return a dict of configuration settings used in the program"""

    conf = {}

    # Determine what dataset to run on. 'mnist', 'cifar10' and 'svhn' are currently supported.
    conf["dataset"] = "svhn"  #'cifar10' #'mnist'
    # Relevant datasets will be put in the location data_root_dir/dataset.
    conf["data_root_dir"] = "/tmp/data"
    # Type of neural net. (CNN adds one conv layer in front)
    conf["net"] = "CNN"
    # Output filename for plot
    conf["out_filename"] = conf["net"] + ".png"  # None

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
    conf["max_steps"] = 20000  # 2000
    # The batch size used in training.
    conf["batch_size"] = 128
    # The step size used by the optimization routine.
    conf["learning_rate"] = 1.0e-2

    # Whether or not to write certain things to stdout.
    conf["verbose"] = True
    # How often (in steps) to log the training progress. Prints to stdout if verbose = True.
    conf["train_progress"] = 10
    # How often (in steps) to evaluate the method on the development partition. Prints to stdout
    # if verbose = True.
    conf["devel_progress"] = 100

    conf["stride"] = None
    conf["pad_size"] = None

    """
    CNN PARAMETERS
    """
    if conf["net"] == "CNN":
        conf["num_filters"] = 3
        conf["height_w"] = 5
        conf["width_w"] = 5
        conf["stride"] = 1
        conf["pad_size"] = 1

    conf["optimizer"] = "adam"

    # parameters for
    adams_dnn = {}
    adams_dnn["first"] = True
    adams_cnn = {}
    adams_cnn["first"] = True

    return conf, adams_dnn, adams_cnn


def plot_progress(conf, train_progress, devel_progress):
    """Plot a chart of the training progress"""

    fig, ax1 = plt.subplots(figsize=(8, 6), dpi=100)
    ax1.plot(
        train_progress["steps"], train_progress["ccr"], color="C0", label="Training set ccr",
    )
    ax1.plot(
        devel_progress["steps"], devel_progress["ccr"], color="C1", label="Development set ccr",
    )
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Correct classification rate")

    ax2 = ax1.twinx()
    ax2.plot(
        train_progress["steps"], train_progress["cost"], color="C2", label="Training set cost",
    )
    ax2.set_ylabel("Cross entropy cost")

    ax1.yaxis.grid(linestyle=":")
    ax2.yaxis.grid(linestyle="--")

    ax1.legend(loc="lower left", bbox_to_anchor=(0, 0.52), framealpha=0.0)
    ax2.legend(loc="lower left", bbox_to_anchor=(0, 0.45), framealpha=0.0)

    plt.title("Training progress")
    fig.tight_layout()

    out_filename = conf["out_filename"]
    if out_filename is not None:
        plt.savefig("../figs" + out_filename)

    plt.show()


def get_data(conf):

    data_dir = os.path.join(conf["data_root_dir"], conf["dataset"])
    print(data_dir)
    if conf["dataset"] == "cifar10":
        conf["channels_x"] = 3
        conf["height_x"] = 32
        conf["width_x"] = 32
        conf["input_dimension"] = conf["channels_x"] * conf["height_x"] * conf["width_x"]
        conf["output_dimension"] = 10
        X_train, Y_train, X_devel, Y_devel, X_test, Y_test = import_data.load_cifar10(data_dir, conf["devel_size"])
    elif conf["dataset"] == "mnist":
        conf["channels_x"] = 1
        conf["height_x"] = 28
        conf["width_x"] = 28
        conf["input_dimension"] = conf["channels_x"] * conf["height_x"] * conf["width_x"]
        conf["output_dimension"] = 10
        X_train, Y_train, X_devel, Y_devel, X_test, Y_test = import_data.load_mnist(data_dir, conf["devel_size"])
    elif conf["dataset"] == "svhn":
        conf["channels_x"] = 3
        conf["height_x"] = 32
        conf["width_x"] = 32
        conf["input_dimension"] = conf["channels_x"] * conf["height_x"] * conf["width_x"]
        conf["output_dimension"] = 10
        X_train, Y_train, X_devel, Y_devel, X_test, Y_test = import_data.load_svhn(data_dir, conf["devel_size"])

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
    conf, adams_dnn, adams_cnn = config()
    X_train, Y_train, X_devel, Y_devel, X_test, Y_test = get_data(conf)
    # Data format is (datasize, channels, height, width) and not DNN flattened yet.
    conf, params_dnn, params_cnn, train_progress, devel_progress = run.train(
        conf, X_train, Y_train, X_devel, Y_devel, adams_dnn, adams_cnn
    )

    plot_progress(conf, train_progress, devel_progress)

    print("Evaluating train set")
    num_correct, num_evaluated = run.evaluate(conf, params_dnn, params_cnn, X_train, Y_train)
    print("CCR = {0:>5} / {1:>5} = {2:>6.4f}".format(num_correct, num_evaluated, num_correct / num_evaluated))
    print("Evaluating development set")
    num_correct, num_evaluated = run.evaluate(conf, params_dnn, params_cnn, X_devel, Y_devel)
    print("CCR = {0:>5} / {1:>5} = {2:>6.4f}".format(num_correct, num_evaluated, num_correct / num_evaluated))
    print("Evaluating test set")
    num_correct, num_evaluated = run.evaluate(conf, params_dnn, params_cnn, X_test, Y_test)
    print("CCR = {0:>5} / {1:>5} = {2:>6.4f}".format(num_correct, num_evaluated, num_correct / num_evaluated))


if __name__ == "__main__":
    main()
