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

    This method stores vectors in dim=-1 before the cdist, and on dim=-3 after
    the cdist. This tensor shaping is inherited from cdist's input and output
    scheme. dim=-3 corresponds to dim=0, except when running in batch mode.
    Ideally this code would use dim=0 and rely on torch.vmap to hide the
    batching complexity, but torch.vmap does not yet work with torch.compile.
    https://github.com/pytorch/pytorch/issues/98822
    """
    state = State(batch_shape)

    ialloc = IndexAllocator()

    lengthscale = vp.symbol("lengthscale")
    x1 = vp.symbol("x1")
    x2 = vp.symbol("x2")

    def index_for_name(name):
        return next(i for i, p in enumerate(space) if p.name == name)

    def scalar_kernel(names):
        ls_indices = ialloc.allocate(len(names))
        indices = torch.tensor([index_for_name(name) for name in names])
        ls = vtorch.unsqueeze(vtorch.index_select(lengthscale, -1, ls_indices),
                              -2)
        return ovt.matern(
            vtorch.cdist(vtorch.index_select(x1, -1, indices)
                         / ls,
                         vtorch.index_select(x2, -1, indices)
                         / ls,
                         p=2),
            nu=2.5)

    def choice_kernel(names):
        ls_indices = ialloc.allocate(len(names))
        indices = torch.tensor([index_for_name(name) for name in names])
        ls = vtorch.unsqueeze(vtorch.index_select(lengthscale, -1, ls_indices),
                              -2)
        return vtorch.exp(
            -vtorch.cdist(vtorch.index_select(x1, -1, indices)
                          / ls,
                          vtorch.index_select(x2, -1, indices)
                          / ls,
                          p=1))


    def scalar_factorized_and_joint(names, suffix):
        # Rather than doing a proper alpha_factorized_or_joint, just weight
        # additive term, so that we can also implement this using gpytorch's
        # built-in capabilities.
        s_factorized = vp.symbol("s_additive" + suffix)
        state.allocate(s_factorized, (),
                       gpytorch.constraints.GreaterThan(1e-4),
                       gpytorch.priors.GammaPrior(2.0, 0.15))

        return vtorch.sum(
            vtorch.stack(
                [vctorch.mul_along_dim(
                    s_factorized,
                    vtorch.sum(
                        vtorch.stack([scalar_kernel([name]) for name in names],
                                     dim=-3),
                        dim=-3),
                    dim=-3),
                 scalar_kernel(names)], dim=-3),
            dim=-3)

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

    regime_kernel = vctorch.fast_prod_positive(
        vtorch.stack(
            [scalar_kernel(regime_joint_names)]
            + regime_kernels("_factorized"),
            dim=-3,
        ),
        dim=-3)
    architecture_kernel = vctorch.fast_prod_positive(
        vtorch.stack(
            [scalar_kernel(architecture_joint_names)]
            + architecture_kernels("_factorized"),
            dim=-3),
        dim=-3)
    joint_kernel = vctorch.fast_prod_positive(
        vtorch.stack(
            [scalar_kernel(regime_joint_names
                           + architecture_joint_names)]
            + regime_kernels("_joint")
            + architecture_kernels("_joint"),
            dim=-3),
        dim=-3)

    scale = vp.symbol("scale")

    state.allocate(scale, (),
                   gpytorch.constraints.GreaterThan(1e-4),
                   gpytorch.priors.GammaPrior(2.0, 0.15))

    # Rather than doing a proper alpha_factorized_or_joint, just weight additive
    # term, so that we can also implement this using gpytorch's built-in
    # capabilities.
    s_factorized = vp.symbol("s_factorized")
    state.allocate(s_factorized, (),
                   gpytorch.constraints.GreaterThan(1e-4),
                   gpytorch.priors.GammaPrior(2.0, 0.15))

    kernel = vctorch.mul_along_dim(
        scale,
        vtorch.sum(
            vtorch.stack(
                [vctorch.mul_along_dim(
                    s_factorized,
                    vtorch.sum(
                        vtorch.stack(
                            [regime_kernel,
                             architecture_kernel],
                            dim=-3),
                        dim=-3),
                    dim=-3),
                 joint_kernel], dim=-3),
            dim=-3),
        dim=-3)

    state.allocate(lengthscale, (ialloc.count,),
                   gpytorch.constraints.GreaterThan(1e-4),
                   gpytorch.priors.GammaPrior(3.0, 6.0))

    return kernel, state.modules


class VexprKernel(gpytorch.kernels.Kernel):
    def __init__(self, kernel_vexpr, state_modules, batch_shape,
                 initialize="mean", vectorize=True, torch_compile=False):
        super().__init__(batch_shape=batch_shape)
        self.kernel_vexpr = kernel_vexpr
        self.state = torch.nn.ModuleDict(state_modules)
        self.kernel_f = None
        self.canary = torch.tensor(0.)
        self.vectorize = vectorize
        self.torch_compile = torch_compile

    def _initialize_from_inputs(self, x1, x2):
        if self.vectorize:
            inputs = {"x1": x1,
                      "x2": x2,
                      **{name: module.value
                         for name, module in self.state.items()}}
            self.kernel_vexpr = vp.vectorize(self.kernel_vexpr, inputs)
            print(self.kernel_vexpr)

        if self.torch_compile:
            kernel_f2 = vp.to_python(self.kernel_vexpr)
        else:
            kernel_f2 = partial(vp.eval, self.kernel_vexpr)

        def kernel_f(x1, x2, parameters):
            return kernel_f2({"x1": x1, "x2": x2, **parameters})

        if self.torch_compile:
            kernel_f = torch.compile(kernel_f)

        self.kernel_f = kernel_f

    def _apply(self, fn):
        self = super()._apply(fn)
        self.canary = fn(self.canary)
        self.kernel_vexpr = tree_map(
            lambda v: (fn(v)
                       if isinstance(v, torch.Tensor)
                       else v),
            self.kernel_vexpr)
        return self

    def forward(self, x1, x2, diag: bool = False, last_dim_is_batch: bool = False):
        assert not diag
        assert not last_dim_is_batch

        parameters = {name: module.value
                      for name, module in self.state.items()}

        with torch.device(self.canary.device):
            if self.kernel_f is None:
                self._initialize_from_inputs(x1, x2)
            return self.kernel_f(x1, x2, parameters)


class VexprPartialHandsOnGP(botorch.models.SingleTaskGP):
    """
    A restricted hands-on model, written to enable apples-to-apples perf
    comparison with the BotorchPartialHandsOnLossModel.
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
                 torch_compile=False):
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
            vectorize=vectorize,
            torch_compile=torch_compile,
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
