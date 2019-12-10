"""Module for training and evaluation"""
import time
import sys
import numpy as np
import functions

np.random.seed(42069)


def get_batch_indices(indices, start_index, end_index):
    """
    Generate batch indexes for data.
    """
    n = len(indices)
    return np.hstack((indices[start_index : min(n, end_index)], indices[0 : max(end_index - n, 0)]))


def train(conf, X_train, Y_train, X_devel, Y_devel):
    """
    Training procedure for network. 
    Takes training and development data and runs through forward and backward propagation.
    Progress is measured with development set at increments chosen in configuration.
    """
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
        # Start timer
        start_time = time.time()

        # Getting indices for batch
        batch_indices = get_batch_indices(example_indices, batch_start_index, batch_end_index)

        # Divide data into batch
        X_batch = X_train[batch_indices]
        Y_batch = functions.one_hot(Y_train[batch_indices], conf["output_dimension"])

        # ---- Forward propagation ----
        # Forward propagation through conv layer
        if conf["net"] == "CNN":
            X_batch, features_cnn = cnn.forward(conf, X_batch, params_cnn, is_training=True)

        # Reshape data for fully connected layer. Flatten.
        X_batch = X_batch.reshape(X_batch.shape[0], -1)
        # Transpose data for fully connected. (DNN expects batch_size as last column)
        X_batch = np.transpose(X_batch, (1, 0))

        # Forward propagation through fully connected classification layers
        Y_proposal, features_dnn = dnn.forward(conf, X_batch, params_dnn, is_training=True)

        # ---- Calculate cost and number of correct predictions ----
        cost_value, num_correct = functions.cross_entropy_cost(Y_proposal, Y_batch)

        # ---- Backpropagation ----
        # Backpropagate through fully connected layers
        grad_params_dnn, dZ = dnn.backward(conf, Y_proposal, Y_batch, params_dnn, features_dnn)
        if conf["net"] == "CNN":
            # Backpropagate through conv layer
            grad_params_cnn = cnn.backward(dZ, params_cnn, params_dnn, conf, features_cnn)

            # ---- Update weights ----
            # Update conv weights
            params_cnn, conf, adams_cnn = functions.optimize(conf, params_cnn, grad_params_cnn, adams_cnn)

        # Update weights in fully connected layer
        params_dnn, conf, adams_dnn = functions.optimize(conf, params_dnn, grad_params_dnn, adams_dnn)

        # Store number of correctly guessed images
        num_correct_since_last_check += num_correct

        # Prepare new batch
        batch_start_index += conf["batch_size"]
        batch_end_index += conf["batch_size"]
        if batch_start_index >= num_examples_in_epoch:
            epoch += 1
            np.random.shuffle(example_indices)
            batch_start_index = 0
            batch_end_index = conf["batch_size"]

        # Step iteration
        step += 1

        # Test if nan is found in costs
        if np.isnan(cost_value):
            print("ERROR: nan encountered")
            break

        # Check progress on training set at increments set in configuration
        if step % conf["train_progress"] == 0:
            # Get time spent
            elapsed_time = time.time() - start_time
            sec_per_batch = elapsed_time / conf["train_progress"]
            examples_per_sec = conf["batch_size"] * conf["train_progress"] / elapsed_time

            # Save accuracy
            ccr = num_correct / conf["batch_size"]
            running_ccr = num_correct_since_last_check / conf["train_progress"] / conf["batch_size"]
            num_correct_since_last_check = 0

            # Save training progress
            train_steps.append(step)
            train_ccr.append(running_ccr)
            train_cost.append(cost_value)

            # Print training progress if verbose=True
            if conf["verbose"]:
                print(
                    "S: {0:>7}, E: {1:>4}, cost: {2:>7.4f}, CCR: {3:>7.4f} ({4:>6.4f}),  "
                    "ex/sec: {5:>7.3e}, sec/batch: {6:>7.3e}".format(
                        step, epoch, cost_value, ccr, running_ccr, examples_per_sec, sec_per_batch,
                    )
                )

        # Check progress against development set
        if step % conf["devel_progress"] == 0:
            # Run development set through evaluation and get predictions.
            num_correct, num_evaluated = evaluate(conf, params_dnn, params_cnn, X_devel, Y_devel)

            # Store development set progress
            devel_steps.append(step)
            devel_ccr.append(num_correct / num_evaluated)

            # Print development progress if verbose=True
            if conf["verbose"]:
                print(
                    "S: {0:>7}, Test on development set. CCR: {1:>5} / {2:>5} = {3:>6.4f}".format(
                        step, num_correct, num_evaluated, num_correct / num_evaluated
                    )
                )

        # Stop training when max steps is reached
        if step >= conf["max_steps"]:
            print("Terminating training after {} steps".format(step))
            break

    # Save progress to dictionaries
    train_progress = {"steps": train_steps, "ccr": train_ccr, "cost": train_cost}
    devel_progress = {"steps": devel_steps, "ccr": devel_ccr}

    return conf, params_dnn, params_cnn, train_progress, devel_progress


def evaluate(conf, params_dnn, params_cnn, X_data, Y_data, output=False):
    """
    Evaluation function for arbitrary data set.
    Takes X and Y data and uses trained weights to make prediction.
    """
    import dnn
    import cnn

    num_examples = X_data.shape[0]
    num_examples_evaluated = 0
    num_correct_total = 0
    start_ind = 0
    end_ind = conf["batch_size"]

    # If output=True, save predictions to file
    if output == True:
        yTrue_out = np.zeros((conf["output_dimension"], Y_data.shape[0]), dtype="int")
        yPred_out = np.zeros((conf["output_dimension"], Y_data.shape[0]), dtype="float")

    # Run evaluation over batches
    while True:
        # Get batches
        X_batch = X_data[start_ind:end_ind]
        Y_batch = functions.one_hot(Y_data[start_ind:end_ind], conf["output_dimension"])

        # ---- Forward propagation ----
        # Forward propagation through conv layer
        if conf["net"] == "CNN":
            X_batch, features_cnn = cnn.forward(conf, X_batch, params_cnn, is_training=True)

        # Reshape data for fully connected layer. Flatten.
        X_batch = X_batch.reshape(X_batch.shape[0], -1)
        # Transpose data for fully connected. (DNN expects batch_size as last column)
        X_batch = np.transpose(X_batch, (1, 0))

        # Forward propagation through fully connected classification layers
        Y_proposal, _ = dnn.forward(conf, X_batch, params_dnn, is_training=False,)

        # ---- Calculate cost and number of correct predictions ----
        _, num_correct = functions.cross_entropy_cost(Y_proposal, Y_batch)

        num_correct_total += num_correct
        num_examples_evaluated += end_ind - start_ind

        # Save batch predictions to output arrays
        if output == True:
            yTrue_out[:, start_ind:end_ind] = Y_batch
            yPred_out[:, start_ind:end_ind] = Y_proposal

        # Chose new batch
        start_ind += conf["batch_size"]
        end_ind += conf["batch_size"]

        # End when all data is checked
        if end_ind >= num_examples:
            end_ind = num_examples
        if start_ind >= num_examples:
            break

    # If output=True, save predictions to file
    if output == True:
        # Format date and time string
        t = time.ctime()
        ta = t.split()
        hms = ta[3].split(":")
        lab = ta[4] + "_" + ta[1] + ta[2] + "_" + hms[0] + hms[1] + hms[2]
        # Save predicitons to file
        np.savetxt("../data/" + conf["output_filename"] + "_true_" + lab + ".dat", (yTrue_out))
        np.savetxt("../data/" + conf["output_filename"] + "_pred_" + lab + ".dat", (yPred_out))

    return num_correct_total, num_examples_evaluated
