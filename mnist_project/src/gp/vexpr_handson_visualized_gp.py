from functools import partial

import botorch
import gpytorch
import outerloop as ol
import outerloop.vexpr.torch as ovt
import torch
import vexpr as vp
import vexpr.torch as vtorch
import vexpr.custom.torch as vctorch

from .gp_utils import (
    StateBuilder,
    IndexAllocator,
    FastStandardize,
    VexprKernel,
    select_divide,
    to_runnable,
    to_visual,
    register_comment_printing,
    print_model_structure,
    print_model_state,
)


N_HOT_PREFIX = "choice_nhot"


def make_handson_kernel(space):
    """
    Creates a kernel vexpr and an object that is ready to instantiate kernel
    parameters, given batch shape.

    This kernel attempts to group parameters into orthogonal groups, while
    also always allowing for the model to learn to use the joint space.

    This method stores vectors in dim=-1 before the cdist, and on dim=-3 after
    the cdist. This tensor shaping is inherited from cdist's input and output
    scheme. dim=-3 corresponds to dim=0, except when running in batch mode.
    Ideally this code would use dim=0 and rely on torch.vmap to hide the
    batching complexity, but torch.vmap does not yet work with torch.compile.
    https://github.com/pytorch/pytorch/issues/98822
    """
    zero_one_exclusive = partial(gpytorch.constraints.Interval,
                                 1e-6,
                                 1 - 1e-6)

    state = StateBuilder()

    ialloc = IndexAllocator()

    lengthscale = vp.symbol("lengthscale")
    x1 = vp.symbol("x1")
    x2 = vp.symbol("x2")

    def index_for_name(name):
        return next(i for i, p in enumerate(space) if p.name == name)

    def scalar_kernel(names):
        ls_indices = ialloc.allocate(len(names))
        ls = vtorch.unsqueeze(vtorch.index_select(lengthscale, -1, ls_indices),
                              -2)
        return ovt.matern(vtorch.cdist(select_divide(names, ls), p=2),
                          nu=2.5)

    def choice_kernel(names):
        ls_indices = ialloc.allocate(len(names))
        ls = vtorch.unsqueeze(vtorch.index_select(lengthscale, -1, ls_indices),
                              -2)
        return vtorch.exp(-vtorch.cdist(select_divide(names, ls),
                                        p=1))

    def scalar_factorized_and_joint(names, suffix):
        w_additive = vp.symbol("w_additive" + suffix)
        alpha_factorized_or_joint = vp.symbol("alpha_factorized_or_joint"
                                              + suffix)
        state.allocate(w_additive, (len(names),),
                       ol.constraints.SoftmaxConstraint(),
                       ol.priors.DirichletPrior(torch.full((len(names),), 2.0)))
        state.allocate(alpha_factorized_or_joint, (1,),
                       zero_one_exclusive(),
                       ol.priors.BetaPrior(5.0, 2.0))

        return vtorch.sum(
            vctorch.mul_along_dim(
                vctorch.heads_tails(alpha_factorized_or_joint),
                vtorch.stack([
                    vtorch.sum(
                        vctorch.mul_along_dim(
                            w_additive,
                            vtorch.stack([scalar_kernel([name])
                                          for name in names],
                                         dim=-3),
                            dim=-3),
                        dim=-3),
                    scalar_kernel(names),
                ], dim=-3),
                dim=-3),
            dim=-3
        )

    def regime_kernels(suffix):
        return [
            # kernel: regime choice parameters
            choice_kernel([f"{N_HOT_PREFIX}{i}"
                           for i in range(4)]),

            # kernel: lr schedule
            scalar_factorized_and_joint(
                ["log_1cycle_initial_lr", "log_1cycle_final_lr",
                 "log_1cycle_max_lr", "log_1cycle_pct_warmup"],
                f"_lr{suffix}"),

            # kernel: momentum schedule
            scalar_factorized_and_joint(
                ["log_1cycle_momentum_max_damping_factor",
                 "log_1cycle_momentum_min_damping_factor",
                 "log_1cycle_beta1_max_damping_factor",
                 "log_1cycle_beta1_min_damping_factor",
                 "log_beta2_damping_factor"],
                f"_momentum{suffix}"),

            # kernel: relative weight decay
            scalar_factorized_and_joint(
                ["log_conv1_wd_div_gmean", "log_conv2_wd_div_gmean",
                 "log_conv3_wd_div_gmean", "log_dense1_wd_div_gmean",
                 "log_dense2_wd_div_gmean"],
                f"_wd{suffix}"),
        ]

    regime_joint_names = ["log_epochs", "log_batch_size",
                          "log_gmean_weight_decay"]

    def architecture_kernels(suffix):
        return [
            # kernel: lr schedule
            scalar_factorized_and_joint(["log_conv1_channels_div_gmean",
                                         "log_conv2_channels_div_gmean",
                                         "log_conv3_channels_div_gmean",
                                         "log_dense1_units_div_gmean"],
                                        f"_units_channels{suffix}"),
        ]

    architecture_joint_names = ["log_gmean_channels_and_units"]

    regime_kernel = vp.with_metadata(
        vtorch.prod(
            vtorch.stack([scalar_kernel(regime_joint_names)]
                         + regime_kernels("_factorized"),
                         dim=-3),
            dim=-3),
        dict(comment="Regime kernel"))
    architecture_kernel = vp.with_metadata(
        vtorch.prod(
            vtorch.stack([scalar_kernel(architecture_joint_names)]
                         + architecture_kernels("_factorized"),
                         dim=-3),
            dim=-3),
        dict(comment="Architecture kernel")
    )
    joint_kernel = vp.with_metadata(
        vtorch.prod(
            vtorch.stack([scalar_kernel(regime_joint_names
                                        + architecture_joint_names)]
                         + regime_kernels("_joint")
                         + architecture_kernels("_joint"),
                         dim=-3),
            dim=-3),
        dict(comment="Joint regime and architecture kernel")
    )

    alpha_regime_vs_architecture = vp.symbol("alpha_regime_vs_architecture")
    state.allocate(alpha_regime_vs_architecture, (1,),
                   zero_one_exclusive(),
                   ol.priors.BetaPrior(2.0, 2.0))
    factorized_kernel = vp.with_metadata(
        vtorch.sum(
            vctorch.mul_along_dim(
                vctorch.heads_tails(
                    alpha_regime_vs_architecture),
                vtorch.stack([
                    regime_kernel,
                    architecture_kernel
                ], dim=-3),
                dim=-3),
            dim=-3),
        dict(comment="Factorized regime and architecture kernels")
    )

    alpha_factorized_vs_joint = vp.symbol("alpha_factorized_vs_joint")
    scale = vp.symbol("scale")

    state.allocate(alpha_factorized_vs_joint, (1,),
                   zero_one_exclusive(),
                   ol.priors.BetaPrior(5.0, 2.0))
    state.allocate(scale, (),
                   gpytorch.constraints.GreaterThan(1e-4),
                   gpytorch.priors.GammaPrior(2.0, 0.15))

    kernel = vctorch.mul_along_dim(
        scale,
        vtorch.sum(
            vctorch.mul_along_dim(
                vctorch.heads_tails(alpha_factorized_vs_joint),
                vtorch.stack([
                    factorized_kernel,
                    joint_kernel
                ], dim=-3),
                dim=-3),
            dim=-3),
        dim=-3)

    state.allocate(lengthscale, (ialloc.count,),
                   gpytorch.constraints.GreaterThan(1e-4),
                   gpytorch.priors.GammaPrior(3.0, 6.0))

    kernel_runnable = vp.bottom_up_transform(partial(to_runnable, index_for_name),
                                    kernel)
    kernel_visualizable = vp.bottom_up_transform(to_visual, kernel)

    return kernel_runnable, kernel_visualizable, state


class VexprHandsOnVisualizedGP(botorch.models.SingleTaskGP):
    """
    Like the VexprHandsOnGP, but written in a way to make the Vexpr
    expression more visualizable.
    """
    def __init__(self, train_X, train_Y,
                 search_space,
                 search_xform,
                 train_Yvar=None,  # included to suppress botorch warnings
                 normalize_input=True,
                 standardize_output=True,
                 # disable when you know all your data is valid to improve
                 # performance (e.g. during cross-validation)
                 round_inputs=True,
                 vectorize=True,
                 torch_compile=False,
                 visualize=True):
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

        (kernel_vexpr,
         kernel_viz_vexpr,
         state_builder) = make_handson_kernel(xform.space2)
        state_modules = state_builder.instantiate(aug_batch_shape)
        covar_module = VexprKernel(
            kernel_vexpr,
            state_modules,
            batch_shape=aug_batch_shape,
            vectorize=vectorize,
            torch_compile=torch_compile,
        )
        self.kernel_viz_vexpr = kernel_viz_vexpr

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
            # noise_prior=gpytorch.priors.GammaPrior(0.9, 10.0),
            noise_prior=gpytorch.priors.GammaPrior(1.1, 0.05),
        )

        self.viz_header_printed = False
        self.visualize = visualize

        register_comment_printing()

        super().__init__(
            train_X, train_Y,
            input_transform=input_transform,
            covar_module=covar_module,
            likelihood=likelihood,
            **extra_kwargs
        )


    def _visualize(self):
        if not self.visualize:
            return

        with torch.no_grad():
            parameters = {name: module.value
                          for name, module in self.covar_module.state.items()}
            viz_expr = vp.partial_eval(self.kernel_viz_vexpr, parameters)

        filename = "handson-fit.txt"

        if not self.viz_header_printed:
            print(f"Logging to {filename}")
            with open(filename, "w") as f:
                print_model_structure(self, f)
            self.viz_header_printed = True

        with open(filename, "a") as f:
            print_model_state(self, f)


    def forward(self, x):
        self._visualize()
        return super().forward(x)
