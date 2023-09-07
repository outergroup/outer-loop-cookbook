"""
Fit a GP to a list of trial directories.
"""

import glob
import os
from functools import partial

import botorch
import gpytorch
import outerloop as ol
import torch


def build_covar_module(space, tree, batch_shape, scale_prior):
    headers = [
        [],
        # Distance (args for sequential.switch), Matern, Prod
        ["parameter_type", "unparsed_tree"],
        # Deeper levels are parsed during sequential.switch
    ]

    operations = [
        # cdist, according to "range" or "choice"
        ol.sequential.switch(
            switch_key="parameter_type",
            value_key="unparsed_tree",
            cases={
                "range": (
                    [
                        # Outer, container level; contains 31 groups
                        [],

                        # Groups; first one contains 3 leaf nodes
                        [],

                        # Nodes
                        ["gamma_prior_args", "key"]],
                    [ol.sequential.select("key", space),
                     ol.sequential.zero_center(),
                     ol.sequential.reweight(
                         ol.sequential.w_lengthscale(
                             gamma_prior_key="gamma_prior_args",
                             constraint=gpytorch.constraints.GreaterThan(1e-04),
                             batch_shape=batch_shape,
                             initialize="mean",
                         ),
                         ndims_per_model=2,
                     ),
                     ol.sequential.cdist_at(),
                     ol.sequential.matern()]
                ),

                "choice": (
                    [[], [],
                     ["gamma_prior_args", "key"]],
                    [ol.sequential.select("key", space),
                     ol.sequential.cdist1d_hamming(),
                     ol.sequential.reweight(
                         ol.sequential.w_lengthscale(
                             gamma_prior_key="gamma_prior_args",
                             constraint=gpytorch.constraints.GreaterThan(1e-06),
                             batch_shape=batch_shape,
                             initialize="mean",
                         ),
                         ndims_per_model=3,
                     ),
                     ol.sequential.mean_at(),
                     ol.sequential.gibbs()])
            }),

        # multiply "range" and "choice"
        ol.sequential.clamp(min=1e-10),
        ol.sequential.prod_at(all_positive=True),
        ol.sequential.reweight(
            ol.sequential.w_scale(
                prior=scale_prior,
                batch_shape=batch_shape,
                constraint=gpytorch.constraints.GreaterThan(1e-6)),
            ndims_per_model=3,
        ),
    ]

    return ol.kernels.KernelFromSequential(
        ol.sequential.build(
            operations,
            ol.treelevels.parse(headers, tree)
        )
    )


class FullyJointLossModel(botorch.models.SingleTaskGP):
    def __init__(self, train_X, train_Y,
                 search_space,
                 search_xform,
                 linear_lengthscale_gamma_prior_args=(3.0, 6.0),
                 log_lengthscale_gamma_prior_args=(3.0, 6.0),
                 cat_lengthscale_gamma_prior_args=(3.0, 6.0),
                 scale_gamma_prior_args=(2.0, 0.15),
                 normalize_input=True,
                 standardize_output=True,
                 # disable when you know all your data is valid to improve
                 # performance (e.g. during cross-validation)
                 round_inputs=True):
        input_batch_shape, aug_batch_shape = self.get_batch_dimensions(
            train_X=train_X, train_Y=train_Y
        )

        xforms = []
        if round_inputs:
            xforms.append(partial(ol.transforms.UntransformThenTransform,
                                  xform=search_xform))
        n_hot_prefix = "choice_nhot"
        xforms.append(partial(ol.transforms.ChoiceNHotProjection,
                              out_name=n_hot_prefix))

        xform = ol.transforms.Chain(search_space, *xforms)

        log_range_names = ["log_epochs", "log_batch_size",
                           "log_conv1_weight_decay", "log_conv2_weight_decay",
                           "log_conv3_weight_decay", "log_dense1_weight_decay",
                           "log_dense2_weight_decay",
                           "log_1cycle_initial_lr_pct",
                           "log_1cycle_final_lr_pct", "log_1cycle_max_lr",
                           "log_conv1_channels", "log_conv2_channels",
                           "log_conv3_channels", "log_dense1_units"]

        linear_range_names = ["1cycle_pct_warmup", "1cycle_max_momentum",
                              "1cycle_min_momentum_pct"]

        choice_names = [f"{n_hot_prefix}{i}"
                        for i in range(4)]

        prior_args_by_name = {}
        for name in linear_range_names:
            prior_args_by_name[name] = linear_lengthscale_gamma_prior_args
        for name in log_range_names:
            prior_args_by_name[name] = log_lengthscale_gamma_prior_args
        for name in choice_names:
            prior_args_by_name[name] = cat_lengthscale_gamma_prior_args

        cont = [
            ("range",
             [(prior_args_by_name[name],
               name,)
              for name in log_range_names]
              + [(prior_args_by_name[name],
                  name,)
                 for name in linear_range_names]),
        ]

        choice = [
            ("choice", [(prior_args_by_name[name], name,)
                        for name in choice_names])
        ]

        joint = cont + choice

        covar_module = build_covar_module(
            xform.space2,
            joint,
            aug_batch_shape,
            scale_prior=gpytorch.priors.GammaPrior(*scale_gamma_prior_args))

        ol.sequential.check(covar_module, xform.space2)

        input_transform = ol.transforms.BotorchInputTransform(xform)
        if normalize_input:
            indices = [i for i, t in enumerate(xform.space2)
                       if n_hot_prefix not in t.name]
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
            extra_kwargs["outcome_transform"] = botorch.models.transforms.Standardize(
                train_Y.shape[-1],
                batch_shape=aug_batch_shape,
            )

        super().__init__(
            train_X, train_Y,
            input_transform=input_transform,
            covar_module=covar_module,
            **extra_kwargs
        )
