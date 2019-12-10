"""Module for training and evaluation"""
import time
import sys
import numpy as np
import functions

np.random.seed(42069)


def get_batch_indices(indices, start_index, end_index):
    n = len(indices)
    return np.hstack((indices[start_index : min(n, end_index)], indices[0 : max(end_index - n, 0)]))


def train(conf, X_train, Y_train, X_devel, Y_devel):
    import dnn
    import cnn

    print("Run training")

    # Preparation
    num_examples_in_epoch = X_train.shape[0]
    example_indices = np.arange(0, num_examples_in_epoch)
    np.random.shuffle(example_indices)

    # Initialisation
    params_dnn, params_cnn = functions.initialization(conf)

    # For displaying training progress
    train_steps = []
    train_ccr = []
    train_cost = []
    devel_steps = []
    devel_ccr = []

    # parameters for adam optimizer
    adams_dnn = {}
    adams_cnn = {}

    # Start training
    step = 0
    epoch = 0
    num_correct_since_last_check = 0
    batch_start_index = 0
    batch_end_index = conf["batch_size"]
    print("Number of training examples in one epoch: ", num_examples_in_epoch)
    print("Start training iteration")
    while True:
        start_time = time.time()
        batch_indices = get_batch_indices(example_indices, batch_start_index, batch_end_index)

        X_batch = X_train[batch_indices]
        X_batch_cnn = X_batch.copy
        Y_batch = functions.one_hot(Y_train[batch_indices], conf["output_dimension"])

        if conf["net"] == "CNN":
            X_batch, features_cnn = cnn.forward(conf, X_batch, params_cnn, is_training=True)

        X_batch = X_batch.reshape(X_batch.shape[0], -1)  # Flatten to dense form
        X_batch = np.transpose(X_batch, (1, 0))  # Reshape for dnn layer

        Y_proposal, features_dnn = dnn.forward(conf, X_batch, params_dnn, is_training=True)

        cost_value, num_correct = functions.cross_entropy_cost(Y_proposal, Y_batch)

        grad_params_dnn, dZ = dnn.backward(conf, Y_proposal, Y_batch, params_dnn, features_dnn)
        if conf["net"] == "CNN":
            grad_params_cnn = cnn.backward(dZ, X_batch_cnn, params_cnn, params_dnn, conf, features_cnn)
            params_cnn, conf, adams_cnn = functions.optimize(conf, params_cnn, grad_params_cnn, adams_cnn)

        params_dnn, conf, adams_dnn = functions.optimize(conf, params_dnn, grad_params_dnn, adams_dnn)

        num_correct_since_last_check += num_correct

        batch_start_index += conf["batch_size"]
        batch_end_index += conf["batch_size"]
        if batch_start_index >= num_examples_in_epoch:
            epoch += 1
            np.random.shuffle(example_indices)
            batch_start_index = 0
            batch_end_index = conf["batch_size"]

        step += 1

        if np.isnan(cost_value):
            print("ERROR: nan encountered")
            break

        if step % conf["train_progress"] == 0:
            elapsed_time = time.time() - start_time
            sec_per_batch = elapsed_time / conf["train_progress"]
            examples_per_sec = conf["batch_size"] * conf["train_progress"] / elapsed_time
            ccr = num_correct / conf["batch_size"]
            running_ccr = num_correct_since_last_check / conf["train_progress"] / conf["batch_size"]
            num_correct_since_last_check = 0
            train_steps.append(step)
            train_ccr.append(running_ccr)
            train_cost.append(cost_value)
            if conf["verbose"]:
                print(
                    "S: {0:>7}, E: {1:>4}, cost: {2:>7.4f}, CCR: {3:>7.4f} ({4:>6.4f}),  "
                    "ex/sec: {5:>7.3e}, sec/batch: {6:>7.3e}".format(
                        step, epoch, cost_value, ccr, running_ccr, examples_per_sec, sec_per_batch,
                    )
                )

        if step % conf["devel_progress"] == 0:
            num_correct, num_evaluated = evaluate(conf, params_dnn, params_cnn, X_devel, Y_devel)
            devel_steps.append(step)
            devel_ccr.append(num_correct / num_evaluated)
            if conf["verbose"]:
                print(
                    "S: {0:>7}, Test on development set. CCR: {1:>5} / {2:>5} = {3:>6.4f}".format(
                        step, num_correct, num_evaluated, num_correct / num_evaluated
                    )
                )

        if step >= conf["max_steps"]:
            print("Terminating training after {} steps".format(step))
            break

    train_progress = {"steps": train_steps, "ccr": train_ccr, "cost": train_cost}
    devel_progress = {"steps": devel_steps, "ccr": devel_ccr}

    return conf, params_dnn, params_cnn, train_progress, devel_progress


def evaluate(conf, params_dnn, params_cnn, X_data, Y_data, output=False):
    import dnn
    import cnn

    num_examples = X_data.shape[0]
    num_examples_evaluated = 0
    num_correct_total = 0
    start_ind = 0
    end_ind = conf["batch_size"]

    if output == True:
        yTrue_out = np.zeros_like(Y_data, dtype="int")
        yPred_out = np.zeros_like(Y_data, dtype="float")

    while True:
        X_batch = X_data[start_ind:end_ind]
        Y_batch = functions.one_hot(Y_data[start_ind:end_ind], conf["output_dimension"])

        if conf["net"] == "CNN":
            X_batch, features_cnn = cnn.forward(conf, X_batch, params_cnn, is_training=True)

        X_batch = X_batch.reshape(X_batch.shape[0], -1)  # Flatten to dense form
        X_batch = np.transpose(X_batch, (1, 0))  # Reshape for dnn layer

        Y_proposal, _ = dnn.forward(conf, X_batch, params_dnn, is_training=False,)
        _, num_correct = functions.cross_entropy_cost(Y_proposal, Y_batch)
        num_correct_total += num_correct

        num_examples_evaluated += end_ind - start_ind

        start_ind += conf["batch_size"]
        end_ind += conf["batch_size"]

        if output == True:
            yTrue_out[start_ind:end_ind] = Y_batch
            yPred_out[start_ind:end_ind] = Y_proposal

        if end_ind >= num_examples:
            end_ind = num_examples

        if start_ind >= num_examples:
            break

    if output == True:
        t = time.ctime()
        ta = t.split()
        hms = ta[3].split(":")
        lab = ta[4] + "_" + ta[1] + ta[2] + "_" + hms[0] + hms[1] + hms[2]

        file1 = open(conf["output_filename"] + "_true_" + lab + ".dat", "w")
        file1.write(yTrue_out)
        file1.close()

        file2 = open(conf["output_filename"] + "_pred_" + lab + ".dat", "w")
        file2.write(yPred_out)
        file2.close()

    return num_correct_total, num_examples_evaluated
