import os
from functools import partial

import outerloop as ol

from .. import gen
from ..trial import mnist_trial1

SMOKE_TEST = os.environ.get("SMOKE_TEST")


max_num_epochs = 60 if not SMOKE_TEST else 5
min_batch_size = 16 if not SMOKE_TEST else 256
space = [
    ol.Choice("optimizer", ["adam", "sgd"]),
    ol.Choice("nesterov", [True, False],
              condition=lambda choices: choices["optimizer"] == "sgd"),
    ol.Int("epochs", 2, max_num_epochs),
    ol.Int("batch_size", min_batch_size, 4096),

    ol.Scalar("conv1_weight_decay", 1e-7, 3e-1),
    ol.Scalar("conv2_weight_decay", 1e-7, 3e-1),
    ol.Scalar("conv3_weight_decay", 1e-7, 3e-1),
    ol.Scalar("dense1_weight_decay", 1e-7, 3e-1),
    ol.Scalar("dense2_weight_decay", 1e-7, 3e-1),

    ol.Scalar("1cycle_initial_lr_pct", 1/80, 1/2),
    ol.Scalar("1cycle_final_lr_pct", 1/30000, 1/100),
    ol.Scalar("1cycle_pct_warmup", 0.01, 0.5),
    ol.Scalar("1cycle_max_lr", 0.01, 20.0),

    # Use "damping factors" (1 - momentum) rather than momentums so that they
    # can be used on a log scale.
    ol.Scalar("1cycle_momentum_max_damping_factor", 1e-4, 1.0,
              condition=lambda choices: choices["optimizer"] == "sgd"),
    ol.Scalar("1cycle_momentum_min_damping_factor_pct", 0.1, 1.0,
              condition=lambda choices: choices["optimizer"] == "sgd"),
    ol.Scalar("1cycle_beta1_max_damping_factor", 1e-4, 0.5,
              condition=lambda choices: choices["optimizer"] == "adam"),
    ol.Scalar("1cycle_beta1_min_damping_factor_pct", 0.1, 1.0,
              condition=lambda choices: choices["optimizer"] == "adam"),
    ol.Scalar("beta2_damping_factor", 1e-4, 0.5,
              condition=lambda choices: choices["optimizer"] == "adam"),

    ol.Int("conv1_channels", 4, 64),
    ol.Int("conv2_channels", 8, 128),
    ol.Int("conv3_channels", 16, 256),
    ol.Int("dense1_units", 8, 256),
]

xform = ol.transforms.ToScalarSpace(
    space,
    ol.transforms.log({
        "epochs": "log_epochs",
        "batch_size": "log_batch_size",
        "1cycle_initial_lr_pct": "log_1cycle_initial_lr_pct",
        "1cycle_final_lr_pct": "log_1cycle_final_lr_pct",
        "1cycle_max_lr": "log_1cycle_max_lr",
        "1cycle_pct_warmup": "log_1cycle_pct_warmup",
        "1cycle_momentum_max_damping_factor": "log_1cycle_momentum_max_damping_factor",
        "1cycle_momentum_min_damping_factor_pct": "log_1cycle_momentum_min_damping_factor_pct",
        "1cycle_beta1_max_damping_factor": "log_1cycle_beta1_max_damping_factor",
        "1cycle_beta1_min_damping_factor_pct": "log_1cycle_beta1_min_damping_factor_pct",
        "beta2_damping_factor": "log_beta2_damping_factor",
        "conv1_weight_decay": "log_conv1_weight_decay",
        "conv2_weight_decay": "log_conv2_weight_decay",
        "conv3_weight_decay": "log_conv3_weight_decay",
        "dense1_weight_decay": "log_dense1_weight_decay",
        "dense2_weight_decay": "log_dense2_weight_decay",
        "conv1_channels": "log_conv1_channels",
        "conv2_channels": "log_conv2_channels",
        "conv3_channels": "log_conv3_channels",
        "dense1_units": "log_dense1_units",
    })
)


mnist1 = dict(
    gen_function=gen.Schedule(
        (512, gen.Sobol(space=space, xform=xform)),
    ),
    trial_function=mnist_trial1.run,
    parameter_space=space,
    search_space=xform.space2,
    search_xform=xform,
)

CONFIGS = dict(
    mnist1=mnist1,
)
