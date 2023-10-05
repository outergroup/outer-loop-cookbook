from functools import partial

import botorch
import gpytorch
import outerloop as ol
import outerloop.vexpr.torch as ovt
import torch
import vexpr as vp
import vexpr.torch as vtorch
import vexpr.custom.torch as vctorch
from jax.tree_util import tree_map

from .gp_utils import State, IndexAllocator, FastStandardize



N_HOT_PREFIX = "choice_nhot"


def make_handson_kernel(space, batch_shape=()):
    """
    This kernel attempts to group parameters into orthogonal groups, while
    also always allowing for the model to learn to use the joint space.
    """
    zero_one_exclusive = partial(gpytorch.constraints.Interval,
                                 1e-6,
                                 1 - 1e-6)

    state = State(batch_shape)

    ialloc = IndexAllocator()

    lengthscale = vp.symbol("lengthscale")
    x1 = vp.symbol("x1")
    x2 = vp.symbol("x2")

    def index_for_name(name):
        return next(i for i, p in enumerate(space) if p.name == name)

    def scalar_kernel(names):
        ls_indices = ialloc.allocate(len(names))
        indices = [index_for_name(name) for name in names]
        return ovt.matern(
            vtorch.cdist(x1[..., indices] / lengthscale[ls_indices],
                         x2[..., indices] / lengthscale[ls_indices],
                         p=2),
            nu=2.5)

    def choice_kernel(names):
        ls_indices = ialloc.allocate(len(names))
        indices = [index_for_name(name) for name in names]
        return ovt.matern(
            vtorch.cdist(x1[..., indices] / lengthscale[ls_indices],
                         x2[..., indices] / lengthscale[ls_indices],
                         p=1),
            nu=2.5)


    def scalar_factorized_and_joint(names, suffix):
        w_additive = vp.symbol("w_additive" + suffix)
        alpha_factorized_or_joint = vp.symbol("alpha_factorized_or_joint"
                                              + suffix)
        state.allocate(w_additive, (len(names),),
                       zero_one_exclusive(),
                       ol.priors.DirichletPrior(torch.full((len(names),), 2.0)))
        state.allocate(alpha_factorized_or_joint, (),
                       zero_one_exclusive(),
                       ol.priors.BetaPrior(4.0, 1.0))
        return vtorch.sum(
            vctorch.heads_tails(alpha_factorized_or_joint)
            * vtorch.stack([
                vtorch.sum(
                    w_additive
                    * vtorch.stack([scalar_kernel([name])
                                    for name in names],
                                   dim=-1),
                    dim=-1),
                scalar_kernel(names),
            ], dim=-1),
            dim=-1
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

    # TODO this might be an unnecessary amount of stacking and using dim=-1. Try
    # using lists, not stack, which implies dim=0. (Currently the vectorization
    # code doesn't gracefully handle mixing dim=-1 in the descendent expressions
    # with dim=0 here.)
    regime_kernel = vctorch.fast_prod_positive(
        vtorch.stack(([scalar_kernel(regime_joint_names)]
                      + regime_kernels("_factorized")),
                     dim=-1),
        dim=-1)
    architecture_kernel = vctorch.fast_prod_positive(
        vtorch.stack(([scalar_kernel(architecture_joint_names)]
                      + architecture_kernels("_factorized")),
                     dim=-1),
        dim=-1)
    joint_kernel = vctorch.fast_prod_positive(
        vtorch.stack(([scalar_kernel(regime_joint_names
                                     + architecture_joint_names)]
                      + regime_kernels("_joint")
                      + architecture_kernels("_joint")),
                     dim=-1),
        dim=-1)

    alpha_regime_vs_architecture = vp.symbol("alpha_regime_vs_architecture")
    alpha_factorized_vs_joint = vp.symbol("alpha_factorized_vs_joint")
    scale = vp.symbol("scale")

    state.allocate(alpha_regime_vs_architecture, (),
                   zero_one_exclusive(),
                   ol.priors.BetaPrior(2.0, 2.0))
    state.allocate(alpha_factorized_vs_joint, (),
                   zero_one_exclusive(),
                   ol.priors.BetaPrior(4.0, 1.0))
    state.allocate(scale, (),
                   gpytorch.constraints.GreaterThan(1e-4),
                   gpytorch.priors.GammaPrior(2.0, 0.15))

    kernel = (scale
              * vtorch.sum(vctorch.heads_tails(alpha_factorized_vs_joint)
                           * vtorch.stack(
                               [vtorch.sum(
                                   vctorch.heads_tails(
                                       alpha_regime_vs_architecture)
                                   * vtorch.stack([
                                       regime_kernel,
                                       architecture_kernel
                                   ], dim=-1),
                                   dim=-1),
                                joint_kernel],
                               dim=-1),
                           dim=-1))

    state.allocate(lengthscale, (ialloc.count,),
                   gpytorch.constraints.GreaterThan(1e-4),
                   gpytorch.priors.GammaPrior(3.0, 6.0))

    return kernel, state.modules


class VexprKernel(gpytorch.kernels.Kernel):
    def __init__(self, kernel_vexpr, state_modules, batch_shape, initialize="mean"):
        super().__init__()
        self.kernel_vexpr = vp.vectorize(kernel_vexpr)
        self.state = torch.nn.ModuleDict(state_modules)

        def kernel_f(x1, x2, parameters):
            return self.kernel_vexpr(x1=x1, x2=x2, **parameters)

        for _ in batch_shape:
            kernel_f = torch.vmap(kernel_f,
                                  in_dims=(0, 0,
                                           {name: 0
                                            for name in state_modules.keys()}))

        self.kernel_f = kernel_f
        self.canary = torch.tensor(0.)

    def _apply(self, fn):
        self = super()._apply(fn)
        self.canary = fn(self.canary)
        self.kernel_vexpr.vexpr = tree_map(
            lambda v: (fn(v)
                       if isinstance(v, torch.Tensor)
                       else v),
            self.kernel_vexpr.vexpr)
        return self

    def forward(self, x1, x2, diag: bool = False, last_dim_is_batch: bool = False):
        assert not diag
        assert not last_dim_is_batch

        parameters = {name: module.value
                      for name, module in self.state.items()}

        with torch.device(self.canary.device):
            return self.kernel_f(x1, x2, parameters)


class VexprHandsOnGP(botorch.models.SingleTaskGP):
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

        covar_module = VexprKernel(
            *make_handson_kernel(xform.space2, aug_batch_shape),
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
