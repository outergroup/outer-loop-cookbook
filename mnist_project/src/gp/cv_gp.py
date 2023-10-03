"""
Fit a GP to a list of trial directories.
"""

import copy
import glob
import os

import botorch
import gpytorch
import outerloop as ol
import torch

from .gp_utils import FastStandardize


class DeterministicFunctionModel(botorch.models.FixedNoiseGP):
    def __init__(self, train_X, train_Y,
                 lengthscale_prior=gpytorch.priors.GammaPrior(3.0, 6.0),
                 outputscale_prior=gpytorch.priors.GammaPrior(2.0, 0.15),
                 constant_noise=1e-4,
                 nu=2.5,
                 **kwargs):
        input_batch_shape, aug_batch_shape = self.get_batch_dimensions(
            train_X=train_X, train_Y=train_Y
        )

        covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=nu,
                ard_num_dims=train_X.shape[-1],
                batch_shape=aug_batch_shape,
                lengthscale_prior=copy.deepcopy(lengthscale_prior),
            ),
            batch_shape=aug_batch_shape,
            outputscale_prior=copy.deepcopy(outputscale_prior),
            # outputscale_constraint=gpytorch.constraints.GreaterThan(1e-4),
        )

        super().__init__(
            train_X, train_Y,
            train_Yvar=torch.full_like(train_Y, constant_noise),
            covar_module=covar_module,
            input_transform=botorch.models.transforms.Normalize(
                train_X.shape[-1],
                batch_shape=aug_batch_shape),
            outcome_transform=FastStandardize(
                train_Y.shape[-1],
                batch_shape=aug_batch_shape,
            **kwargs
        ))


class NoisyModel(botorch.models.SingleTaskGP):
    def __init__(self, train_X, train_Y,
                 lengthscale_prior=gpytorch.priors.GammaPrior(3.0, 6.0),
                 outputscale_prior=gpytorch.priors.GammaPrior(2.0, 0.15),
                 noise_prior=gpytorch.priors.GammaPrior(1.1, 0.05),
                 **kwargs):
        input_batch_shape, aug_batch_shape = self.get_batch_dimensions(
            train_X=train_X, train_Y=train_Y
        )

        covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=2.5,
                ard_num_dims=train_X.shape[-1],
                batch_shape=aug_batch_shape,
                lengthscale_prior=copy.deepcopy(lengthscale_prior),
            ),
            batch_shape=aug_batch_shape,
            outputscale_prior=copy.deepcopy(outputscale_prior),
        )

        if noise_prior is None:
            initial_noise = gpytorch.priors.GammaPrior(1.1, 0.05).mode
        else:
            initial_noise = noise_prior.mode

        likelihood = gpytorch.likelihoods.GaussianLikelihood(
                noise_prior=noise_prior,
                batch_shape=aug_batch_shape,
                noise_constraint=gpytorch.constraints.GreaterThan(
                    botorch.models.gp_regression.MIN_INFERRED_NOISE_LEVEL,
                    transform=None,
                    initial_value=initial_noise,
                )
        )

        super().__init__(
            train_X, train_Y,
            covar_module=covar_module,
            likelihood=likelihood,
            input_transform=botorch.models.transforms.Normalize(
                train_X.shape[-1],
                batch_shape=aug_batch_shape),
            outcome_transform=botorch.models.transforms.Standardize(
                train_Y.shape[-1],
                batch_shape=aug_batch_shape,
            **kwargs
        ))
