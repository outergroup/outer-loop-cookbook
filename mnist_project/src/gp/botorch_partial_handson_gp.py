from functools import partial

import botorch
import gpytorch
import outerloop as ol
import torch

from .gp_utils import FastStandardize


N_HOT_PREFIX = "choice_nhot"


def make_handson_kernel(space, batch_shape=()):
    """
    This kernel attempts to group parameters into orthogonal groups, while
    also always allowing for the model to learn to use the joint space.
    """
    def index_for_name(name):
        return next(i for i, p in enumerate(space) if p.name == name)

    def scalar_kernel(names):
        return gpytorch.kernels.MaternKernel(
            nu=2.5,
            batch_shape=batch_shape,
            ard_num_dims=len(names),
            active_dims=[index_for_name(name) for name in names],
            lengthscale_constraint=gpytorch.constraints.GreaterThan(1e-04),
            lengthscale_prior=gpytorch.priors.GammaPrior(3.0, 6.0),
        )

    def choice_kernel(names):
        return botorch.models.kernels.CategoricalKernel(
            batch_shape=batch_shape,
            ard_num_dims=len(names),
            active_dims=[index_for_name(name) for name in names],
            lengthscale_constraint=gpytorch.constraints.GreaterThan(1e-06),
            lengthscale_prior=gpytorch.priors.GammaPrior(3.0, 6.0),
        )

    def scalar_factorized_and_joint(names):
        fast_additive = gpytorch.kernels.AdditiveStructureKernel(
            gpytorch.kernels.MaternKernel(
                nu=2.5,
                ard_num_dims=len(names),
                lengthscale_constraint=gpytorch.constraints.GreaterThan(1e-04),
                lengthscale_prior=gpytorch.priors.GammaPrior(3.0, 6.0),
            ),
            num_dims=len(names),
            active_dims=[index_for_name(name) for name in names]
        )
        joint = scalar_kernel(names)

        return gpytorch.kernels.ScaleKernel(
            fast_additive,
            outputscale_constraint=gpytorch.constraints.GreaterThan(1e-4),
            outputscale_prior=gpytorch.priors.GammaPrior(2.0, 0.15),
        ) + joint

    def regime_kernels():
        return [
            # kernel: regime choice parameters
            choice_kernel([f"{N_HOT_PREFIX}{i}"
                           for i in range(4)]),

            # kernel: lr schedule
            scalar_factorized_and_joint(
                ["log_1cycle_initial_lr", "log_1cycle_final_lr",
                 "log_1cycle_max_lr", "log_1cycle_pct_warmup"]),

            # kernel: momentum schedule
            scalar_factorized_and_joint(
                ["log_1cycle_momentum_max_damping_factor",
                 "log_1cycle_momentum_min_damping_factor",
                 "log_1cycle_beta1_max_damping_factor",
                 "log_1cycle_beta1_min_damping_factor",
                 "log_beta2_damping_factor"]),

            # kernel: relative weight decay
            scalar_factorized_and_joint(
                ["log_conv1_wd_div_gmean", "log_conv2_wd_div_gmean",
                 "log_conv3_wd_div_gmean", "log_dense1_wd_div_gmean",
                 "log_dense2_wd_div_gmean"]),
        ]

    regime_joint_names = ["log_epochs", "log_batch_size",
                          "log_gmean_weight_decay"]

    def architecture_kernels():
        return [
            # kernel: lr schedule
            scalar_factorized_and_joint(["log_conv1_channels_div_gmean",
                                         "log_conv2_channels_div_gmean",
                                         "log_conv3_channels_div_gmean",
                                         "log_dense1_units_div_gmean"]),
        ]

    architecture_joint_names = ["log_gmean_channels_and_units"]

    regime_kernel = gpytorch.kernels.ProductKernel(
        *([scalar_kernel(regime_joint_names)]
          + regime_kernels()))
    architecture_kernel = gpytorch.kernels.ProductKernel(
        *([scalar_kernel(architecture_joint_names)]
          + architecture_kernels()))
    joint_kernel = gpytorch.kernels.ProductKernel(
        *([scalar_kernel(regime_joint_names + architecture_joint_names)]
          + regime_kernels()
          + architecture_kernels()))

    factorized_kernel = gpytorch.kernels.ScaleKernel(
        regime_kernel + architecture_kernel,
        outputscale_constraint=gpytorch.constraints.GreaterThan(1e-4),
        outputscale_prior=gpytorch.priors.GammaPrior(2.0, 0.15),
    )

    kernel = gpytorch.kernels.ScaleKernel(
        factorized_kernel + joint_kernel,
        outputscale_constraint=gpytorch.constraints.GreaterThan(1e-4),
        outputscale_prior=gpytorch.priors.GammaPrior(2.0, 0.15),
    )

    return kernel


class BotorchPartialHandsOnGP(botorch.models.SingleTaskGP):
    """
    Attempt to build a "hands-on" kernel using gpytorch and botorch's built-in
    kernels.
    """
    def __init__(self, train_X, train_Y,
                 search_space,
                 search_xform,
                 train_Yvar=None,  # included to suppress botorch warnings
                 normalize_input=True,
                 standardize_output=True,
                 # disable when you know all your data is valid to improve
                 # performance (e.g. during cross-validation)
                 round_inputs=True):
        assert train_Yvar is None
        input_batch_shape, aug_batch_shape = self.get_batch_dimensions(
            train_X=train_X, train_Y=train_Y
        )

        xforms = []
        if round_inputs:
            xforms.append(partial(ol.transforms.UntransformThenTransform,
                                  xform=search_xform))

        xforms += [
            ol.transforms.append_mean(
                ["log_conv1_channels", "log_conv2_channels",
                 "log_conv3_channels", "log_dense1_units"],
                "log_gmean_channels_and_units"),
            ol.transforms.subtract(
                {"log_conv1_channels": "log_conv1_channels_div_gmean",
                 "log_conv2_channels": "log_conv2_channels_div_gmean",
                 "log_conv3_channels": "log_conv3_channels_div_gmean",
                 "log_dense1_units": "log_dense1_units_div_gmean"},
                "log_gmean_channels_and_units"),
            ol.transforms.add(
                {"log_1cycle_initial_lr_pct": "log_1cycle_initial_lr",
                 "log_1cycle_final_lr_pct": "log_1cycle_final_lr"},
                "log_1cycle_max_lr"),
            ol.transforms.add(
                {"log_1cycle_momentum_min_damping_factor_pct":
                 "log_1cycle_momentum_min_damping_factor"},
                "log_1cycle_momentum_max_damping_factor"),
            ol.transforms.add(
                {"log_1cycle_beta1_min_damping_factor_pct":
                 "log_1cycle_beta1_min_damping_factor"},
                "log_1cycle_beta1_max_damping_factor"),
            ol.transforms.append_mean(
                ["log_conv1_weight_decay", "log_conv2_weight_decay",
                 "log_conv3_weight_decay", "log_dense1_weight_decay",
                 "log_dense2_weight_decay"],
                "log_gmean_weight_decay"),
            ol.transforms.subtract(
                {"log_conv1_weight_decay": "log_conv1_wd_div_gmean",
                 "log_conv2_weight_decay": "log_conv2_wd_div_gmean",
                 "log_conv3_weight_decay": "log_conv3_wd_div_gmean",
                 "log_dense1_weight_decay": "log_dense1_wd_div_gmean",
                 "log_dense2_weight_decay": "log_dense2_wd_div_gmean"},
                "log_gmean_weight_decay"),
            partial(ol.transforms.ChoiceNHotProjection,
                    out_name=N_HOT_PREFIX)
        ]

        xform = ol.transforms.Chain(search_space, *xforms)

        covar_module = make_handson_kernel(xform.space2, aug_batch_shape)

        input_transform = ol.transforms.BotorchInputTransform(xform)
        if normalize_input:
            indices = [i for i, t in enumerate(xform.space2)
                       if N_HOT_PREFIX not in t.name]
            input_transform = botorch.models.transforms.ChainedInputTransform(
                main=input_transform,
                normalize=botorch.models.transforms.Normalize(
                    len(indices),
                    indices=indices,
                    batch_shape=aug_batch_shape,
                ),
            )

        extra_kwargs = {}
        if standardize_output:
            extra_kwargs["outcome_transform"] = FastStandardize(
                train_Y.shape[-1],
                batch_shape=aug_batch_shape,
            )

        # Use likelihood from botorch MixedSingleTaskGP
        min_noise = 1e-5 if train_X.dtype == torch.float else 1e-6
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            batch_shape=aug_batch_shape,
            noise_constraint=gpytorch.constraints.GreaterThan(
                min_noise, transform=None, initial_value=1e-3
            ),
            noise_prior=gpytorch.priors.GammaPrior(0.9, 10.0),
        )

        super().__init__(
            train_X, train_Y,
            input_transform=input_transform,
            covar_module=covar_module,
            likelihood=likelihood,
            **extra_kwargs
        )
