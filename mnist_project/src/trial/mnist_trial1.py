"""
Trial: Train MNIST, given a set of LeNet hyperparameters
"""

import json
import math
import os
import time


SMOKE_TEST = os.environ.get("SMOKE_TEST")

MAX_TRAINING_TIME = 30 if not SMOKE_TEST else 10
EARLY_STOP_TIME_THRESHOLD = 40 if not SMOKE_TEST else 20
LOSS_MAX = 20.0

WARMED_UP = [False]

def evaluate(config, verbose=0):
    """
    A simple LeNet MNIST on TensorFlow task.

    I chose TensorFlow because it runs quickly on the M1 MacBook Pro with
    tensorflow-metal (which is buggy, frequently locking up, but I find it works
    well with this code if I use old version 0.5.0 and tensorflow-macos 2.9.2)
    """
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    import tensorflow as tf
    # Import keras like this so that pylance doesn't get confused.
    keras = tf.keras
    from keras import layers, models, regularizers
    from keras.datasets import mnist

    # print("Listing GPUs:")
    # print(tf.config.list_physical_devices('GPU'))

    # print CUDNN_PATH environment variable
    # print("CUDNN_PATH:", os.environ.get("CUDNN_PATH"))
    # print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH"))

    (train_images,
     train_labels), (test_images,
                     test_labels) = mnist.load_data()
    train_images = train_images.reshape((60000, 28, 28, 1))
    train_images = train_images.astype("float32") / 255
    test_images = test_images.reshape((10000, 28, 28, 1))
    test_images = test_images.astype("float32") / 255

    model = models.Sequential()
    model.add(layers.Conv2D(
        config["conv1_channels"], (3, 3), activation="relu",
        input_shape=(28, 28, 1),
        kernel_regularizer=regularizers.L2(config["conv1_weight_decay"])
    ))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(
        config["conv2_channels"], (3, 3), activation="relu",
        kernel_regularizer=regularizers.L2(config["conv2_weight_decay"])
    ))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(
        config["conv3_channels"], (3, 3), activation="relu",
        kernel_regularizer=regularizers.L2(config["conv3_weight_decay"])
    ))
    model.add(layers.Flatten())
    model.add(layers.Dense(
        config["dense1_units"],
        activation="relu",
        kernel_regularizer=regularizers.L2(config["dense1_weight_decay"])
    ))
    model.add(layers.Dense(10, activation="softmax",
                            kernel_regularizer=regularizers.L2(config["dense2_weight_decay"])
                            ))
    # print(model.summary())

    optimizers = {
        "adam": keras.optimizers.legacy.Adam,
        "sgd": keras.optimizers.legacy.SGD,
        "rmsprop": keras.optimizers.RMSprop,
    }

    optimizer_args = {}
    if config["optimizer"] in ("sgd", "rmsprop"):
        optimizer_args["momentum"] = config["1cycle_max_momentum"]
    if config["optimizer"] == "sgd":
        optimizer_args["nesterov"] = config["nesterov"]

    model.compile(
        optimizer=optimizers[config["optimizer"]](
            learning_rate=(config["1cycle_initial_lr_pct"]
                            * config["1cycle_max_lr"]),
            **optimizer_args
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"])

    batches_per_epoch = int(math.ceil(train_images.shape[0]
                                        / config["batch_size"]))
    total_batches = batches_per_epoch * config["epochs"]

    class OneCycleLR(keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.batch_idx = 0

        def on_train_batch_begin(self, batch, logs=None):
            warmup_pct = config["1cycle_pct_warmup"]
            set_momentum = config["optimizer"] == "sgd"

            if (self.batch_idx / total_batches) < warmup_pct:
                pct = self.batch_idx / (warmup_pct * total_batches)
                lr2 = config["1cycle_max_lr"]
                lr1 = config["1cycle_initial_lr_pct"] * lr2

                if set_momentum:
                    momentum1 = config["1cycle_max_momentum"]
                    momentum2 = config["1cycle_min_momentum_pct"] * momentum1
            else:
                pct = ((self.batch_idx - (warmup_pct * total_batches))
                        / ((1 - warmup_pct) * total_batches))
                lr1 = config["1cycle_max_lr"]
                lr2 = config["1cycle_final_lr_pct"] * lr1

                if set_momentum:
                    momentum2 = config["1cycle_max_momentum"]
                    momentum1 = config["1cycle_min_momentum_pct"] * momentum2

            keras.backend.set_value(
                self.model.optimizer.lr,
                lr1 + pct * (lr2 - lr1))

            if set_momentum:
                keras.backend.set_value(
                    self.model.optimizer.momentum,
                    momentum1 + pct * (momentum2 - momentum1))

        def on_train_batch_end(self, batch, logs=None):
            self.batch_idx += 1

    early_stopped = [False]
    estimated_total_duration = [None]
    tstart = time.time()

    class EarlyStopLongExperiments(keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.epoch_start_time = None

        def on_epoch_begin(self, epoch, logs={}):
            self.epoch_start_time = time.time()

        def on_epoch_end(self, epoch, logs={}):
            global WARMED_UP
            epoch_duration = time.time() - self.epoch_start_time

            if (WARMED_UP[0]
                and epoch == 0 or not WARMED_UP[0] and epoch == 1):
                WARMED_UP[0] = True

                estimated_total_duration[0] = epoch_duration * config["epochs"]
                if estimated_total_duration[0] > EARLY_STOP_TIME_THRESHOLD:
                    self.model.stop_training = True
                    early_stopped[0] = True

    model.fit(train_images, train_labels,
                epochs=config["epochs"],
                batch_size=config["batch_size"],
                callbacks=[
                    OneCycleLR(),
                    EarlyStopLongExperiments(),
                ],
                verbose=verbose
                )

    training_time = time.time() - tstart

    if early_stopped[0]:
        if verbose > 0:
            print("Early stopping due to long estimated training time")
        return {"training_time": estimated_total_duration[0]}
    else:
        test_loss, test_acc = model.evaluate(test_images,
                                             test_labels,
                                             verbose=0)

        if (test_loss == math.inf
            or math.isnan(test_loss)
            or test_loss > LOSS_MAX):
            test_loss = LOSS_MAX

        return {"accuracy": test_acc,
                "loss": test_loss,
                "training_time": training_time}


def run(trial_dir, args, verbose=0):
    result = evaluate(args, verbose=verbose)
    print(result)

    if trial_dir is not None:
        result_path = os.path.join(trial_dir, "result.json")
        with open(result_path, "w") as f:
            print("Saving", result_path)
            json.dump(result, f)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-dir", type=str, default=None)
    args = parser.parse_args()

    config = {
        "optimizer": "adam", "epochs": 6, "batch_size": 1360,
        "conv1_weight_decay": 0.00033745087809545213, "conv2_weight_decay":
        0.00016976241014781247, "conv3_weight_decay": 1.5217237815361623e-06,
        "dense1_weight_decay": 2.077255561622722e-05, "dense2_weight_decay":
        3.080133955890812e-05, "1cycle_initial_lr_pct": 0.10123216560356099,
        "1cycle_final_lr_pct": 0.0006318452592530336, "1cycle_pct_warmup":
        0.364258366888389, "1cycle_max_lr": 0.08023546878117822,
        "conv1_channels": 12, "conv2_channels": 11, "conv3_channels": 49,
        "dense1_units": 115
    }

    if args.trial_dir is not None:
        os.makedirs(args.trial_dir, exist_ok=True)

    run(args.trial_dir, config)
