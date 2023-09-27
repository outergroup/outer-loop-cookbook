from functools import partial

import botorch
import gpytorch
import outerloop as ol
import torch
from botorch.models.kernels.categorical import CategoricalKernel
from gpytorch.constraints import GreaterThan
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.priors import GammaPrior

N_HOT_PREFIX = "choice_nhot"


class BotorchMixedGP(botorch.models.MixedSingleTaskGP):
    """
    An implementation of botorch.models.MixedSingleTaskGP with the following changes:

    - Allows input transforms to change the length of X.
    - Assigns priors to parameters
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
            partial(ol.transforms.ChoiceNHotProjection, out_name=N_HOT_PREFIX)
        ]

        xform = ol.transforms.Chain(search_space, *xforms)

        range_indices = []
        nhot_indices = []
        for i, p in enumerate(xform.space2):
            if N_HOT_PREFIX in p.name:
                nhot_indices.append(i)
            else:
                range_indices.append(i)

        input_transform = ol.transforms.BotorchInputTransform(xform)
        if normalize_input:
            input_transform = botorch.models.transforms.ChainedInputTransform(
                main=input_transform,
                normalize=botorch.models.transforms.Normalize(
                    len(range_indices),
                    indices=range_indices,
                    batch_shape=aug_batch_shape,
                ),
            )

        extra_kwargs = {}
        if standardize_output:
            extra_kwargs["outcome_transform"] = botorch.models.transforms.Standardize(
                train_Y.shape[-1],
                batch_shape=aug_batch_shape,
            )

        ord_dims = range_indices
        cat_dims = nhot_indices

        # Begin code adapted from MixedSingleTaskGP

        if len(cat_dims) == 0:
            raise ValueError(
                "Must specify categorical dimensions for MixedSingleTaskGP"
            )
        self._ignore_X_dims_scaling_check = cat_dims

        def cont_kernel_factory(
            batch_shape,
            ard_num_dims,
            active_dims,
        ) -> MaternKernel:
            return MaternKernel(
                nu=2.5,
                batch_shape=batch_shape,
                ard_num_dims=ard_num_dims,
                active_dims=active_dims,
                lengthscale_constraint=GreaterThan(1e-04),
                lengthscale_prior=gpytorch.priors.GammaPrior(3.0, 6.0),
            )

        # This Gamma prior is quite close to the Horseshoe prior
        min_noise = 1e-5 if train_X.dtype == torch.float else 1e-6
        likelihood = GaussianLikelihood(
            batch_shape=aug_batch_shape,
            noise_constraint=GreaterThan(
                min_noise, transform=None, initial_value=1e-3
            ),
            noise_prior=GammaPrior(0.9, 10.0),
        )

        d = train_X.shape[-1]
        if len(ord_dims) == 0:
            covar_module = ScaleKernel(
                CategoricalKernel(
                    batch_shape=aug_batch_shape,
                    ard_num_dims=len(cat_dims),
                    lengthscale_constraint=GreaterThan(1e-06),
                ),
                outputscale_constraint=gpytorch.constraints.GreaterThan(1e-4),
                outputscale_prior=gpytorch.priors.GammaPrior(2.0, 0.15)
            )
        else:
            sum_kernel = ScaleKernel(
                cont_kernel_factory(
                    batch_shape=aug_batch_shape,
                    ard_num_dims=len(ord_dims),
                    active_dims=ord_dims,
                )
                + ScaleKernel(
                    CategoricalKernel(
                        batch_shape=aug_batch_shape,
                        ard_num_dims=len(cat_dims),
                        active_dims=cat_dims,
                        lengthscale_constraint=GreaterThan(1e-06),
                        lengthscale_prior=gpytorch.priors.GammaPrior(3.0, 6.0),
                    )
                ),
                outputscale_constraint=gpytorch.constraints.GreaterThan(1e-4),
                outputscale_prior=gpytorch.priors.GammaPrior(2.0, 0.15)
            )
            prod_kernel = ScaleKernel(
                cont_kernel_factory(
                    batch_shape=aug_batch_shape,
                    ard_num_dims=len(ord_dims),
                    active_dims=ord_dims,
                )
                * CategoricalKernel(
                    batch_shape=aug_batch_shape,
                    ard_num_dims=len(cat_dims),
                    active_dims=cat_dims,
                    lengthscale_constraint=GreaterThan(1e-06),
                    lengthscale_prior=gpytorch.priors.GammaPrior(3.0, 6.0),
                ),
                outputscale_constraint=gpytorch.constraints.GreaterThan(1e-4),
                outputscale_prior=gpytorch.priors.GammaPrior(2.0, 0.15)
            )
            covar_module = sum_kernel + prod_kernel

        super(botorch.models.MixedSingleTaskGP, self).__init__(
            train_X=train_X,
            train_Y=train_Y,
            likelihood=likelihood,
            covar_module=covar_module,
            input_transform=input_transform,
            **extra_kwargs
        )
