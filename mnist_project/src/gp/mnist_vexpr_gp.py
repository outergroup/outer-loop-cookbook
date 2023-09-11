"""
Fit a GP to a list of trial directories.
"""

import math
from functools import partial

import botorch
import gpytorch
import outerloop as ol
import outerloop.vexpr.torch as ovt
import torch

import vexpr as vp
import vexpr.torch as vtorch


N_HOT_PREFIX = "choice_nhot"


class VexprKernel(gpytorch.kernels.Kernel):
    def __init__(self,
                 space,
                 linear_lengthscale_gamma_prior_args,
                 log_lengthscale_gamma_prior_args,
                 cat_lengthscale_gamma_prior_args,
                 scale_gamma_prior_args,
                 scale_constraint=gpytorch.constraints.GreaterThan(1e-06),
                 lengthscale_constraint=gpytorch.constraints.GreaterThan(1e-06),
                 initialize="mean",
                 batch_shape=torch.Size([])):
        super().__init__()

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

        choice_names = [f"{N_HOT_PREFIX}{i}"
                        for i in range(4)]

        scale_prior = gpytorch.priors.GammaPrior(*scale_gamma_prior_args)

        scale_constraint = gpytorch.constraints.GreaterThan(1e-06)
        name = "raw_scale"
        self.register_parameter(
            name=name,
            parameter=torch.nn.Parameter(
                torch.zeros(*batch_shape, 1))
        )

        if scale_constraint is None:
            scale_constraint = gpytorch.constraints.Positive()

        self.register_constraint(name, scale_constraint)

        if scale_prior is not None:
            self.register_prior(f"scale_prior", scale_prior,
                                VexprKernel.get_scale,
                                VexprKernel.set_scale)


        n = len(log_range_names) + len(linear_range_names) + len(choice_names)
        concentration_rate = []
        for i in range(n):
            # this kernel assumes one lengthscale per input feature.
            name = space[i].name
            if name in log_range_names:
                concentration_rate.append(log_lengthscale_gamma_prior_args)
            elif name in linear_range_names:
                concentration_rate.append(linear_lengthscale_gamma_prior_args)
            elif name in choice_names:
                concentration_rate.append(cat_lengthscale_gamma_prior_args)
            else:
                raise ValueError(name)
        concentration, rate = zip(*concentration_rate)
        concentration = torch.tensor(concentration)
        rate = torch.tensor(rate)
        lengthscale_gamma_prior_args = concentration, rate
        lengthscale_prior = gpytorch.priors.GammaPrior(*lengthscale_gamma_prior_args)
        name = "raw_lengthscale"
        self.register_parameter(
            name=name,
            parameter=torch.nn.Parameter(
                torch.zeros(*batch_shape, n))
        )

        if lengthscale_constraint is None:
            lengthscale_constraint = gpytorch.constraints.Positive()

        self.register_constraint(name, lengthscale_constraint)

        if lengthscale_prior is not None:
            self.register_prior(f"lengthscale_prior", lengthscale_prior,
                                VexprKernel.get_lengthscale,
                                VexprKernel.set_lengthscale)
            if initialize is not None:
                if initialize == "mode":
                    value = lengthscale_prior.mode
                elif initialize == "mean":
                    value = lengthscale_prior.mean
                else:
                    raise ValueError(f"Unrecognized initialization: {initialize}")

                self.set_lengthscale(self, value)

        (cont_indices,
         cat_indices) = ([next(i for i, t in enumerate(space)
                               if t.name == name)
                          for name in names]
                         for names in (log_range_names + linear_range_names,
                                       choice_names))
        cont_indices = torch.tensor(cont_indices)
        cat_indices = torch.tensor(cat_indices)

        @vp.vectorize
        def kernel_f(x1, x2, scale, lengthscale):
            return (scale
                    * vtorch.prod([
                        ovt.matern(vtorch.cdist(x1[..., cont_indices]
                                                / lengthscale[cont_indices],
                                                x2[..., cont_indices]
                                                / lengthscale[cont_indices],
                                                p=2),
                                   nu=2.5),
                        ovt.matern(vtorch.cdist(x1[..., cat_indices]
                                                / lengthscale[cat_indices],
                                                x2[..., cat_indices]
                                                / lengthscale[cat_indices],
                                                p=1))
                        # vtorch.exp(-vtorch.cdist(x1[..., cat_indices]
                        #                          / lengthscale[cat_indices],
                        #                          x2[..., cat_indices]
                        #                          / lengthscale[cat_indices],
                        #                          p=1))
                    ], dim=0))

        # Enable batching
        for _ in batch_shape:
            kernel_f = torch.vmap(kernel_f, in_dims=(0, 0, 0, 0))

        self.kernel_f = kernel_f

    def forward(self, x1, x2, diag: bool = False, last_dim_is_batch: bool = False):
        assert not diag
        assert not last_dim_is_batch

        d = self.kernel_f(x1, x2, self.scale, self.lengthscale)
        return d

    # def _apply(self, fn):
    #     self = super()._apply(self, fn)
    #     # TODO: this causes bugs by converting primitives into new instances
    #     # of primitives, breaking "is symbol_p" checks
    #     self.kernel_f = tree_map(
    #         lambda v: (fn(v)
    #                    if isinstance(v, torch.Tensor)
    #                    else v),
    #         self.kernel_f)
    #     return self

    @property
    def lengthscale(self):
        with torch.profiler.record_function("VexprKernel.lengthscale"):
            if self.training:
                return self.raw_lengthscale_constraint.transform(self.raw_lengthscale)
            else:
                with torch.no_grad():
                    return self.raw_lengthscale_constraint.transform(self.raw_lengthscale)

    @staticmethod
    def get_lengthscale(instance):
        return instance.lengthscale

    @staticmethod
    def set_lengthscale(instance, v):
        if not torch.is_tensor(v):
            v = torch.as_tensor(v).to(instance.raw_lengthscale)

        instance.initialize(
            **{"raw_lengthscale":
               instance.raw_lengthscale_constraint.inverse_transform(v)}
        )

    @property
    def scale(self):
        with torch.profiler.record_function("VexprKernel.scale"):
            if self.training:
                return self.raw_scale_constraint.transform(self.raw_scale)
            else:
                with torch.no_grad():
                    return self.raw_scale_constraint.transform(self.raw_scale)

    @staticmethod
    def get_scale(instance):
        return instance.scale

    @staticmethod
    def set_scale(instance, v):
        if not torch.is_tensor(v):
            v = torch.as_tensor(v).to(instance.raw_scale)

        instance.initialize(
            **{"raw_scale":
               instance.raw_scale_constraint.inverse_transform(v)}
        )


class VexprFullyJointLossModel(botorch.models.SingleTaskGP):
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
        xforms.append(partial(ol.transforms.ChoiceNHotProjection,
                              out_name=N_HOT_PREFIX))

        xform = ol.transforms.Chain(search_space, *xforms)

        covar_module = VexprKernel(
            space=xform.space2,
            linear_lengthscale_gamma_prior_args=linear_lengthscale_gamma_prior_args,
            log_lengthscale_gamma_prior_args=log_lengthscale_gamma_prior_args,
            cat_lengthscale_gamma_prior_args=cat_lengthscale_gamma_prior_args,
            scale_gamma_prior_args=scale_gamma_prior_args,
            batch_shape=aug_batch_shape,
        )

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
