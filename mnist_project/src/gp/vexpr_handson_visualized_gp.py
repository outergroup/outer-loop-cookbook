from functools import partial

import botorch
import gpytorch
import outerloop as ol
import outerloop.vexpr.torch as ovt
import torch
import vexpr as vp
import vexpr.torch as vtorch
import vexpr.custom.torch as vctorch
from torch.utils._pytree import tree_map
from vexpr.core import _p_and_constructor

from .gp_utils import (
    StateBuilder,
    IndexAllocator,
    FastStandardize,
)


# Vexpr primitives that never get run. Some get translated into runnable
# primitives, others are intended for visualization.
select_divide_p, select_divide = _p_and_constructor("select_divide")
select_p, select = _p_and_constructor("select")


def to_runnable(index_for_name, expr):
    if expr.op == vtorch.primitives.cdist_p:
        # for any expr that has a select_divide as a child, convert it to a pair
        # of arguments
        assert expr.args[0].op == select_divide_p
        names, ls = expr.args[0].args

        indices = torch.tensor([index_for_name(name) for name in names])
        expr = expr.update_args([
            vtorch.index_select(vp.symbol("x1"), -1, indices) / ls,
            vtorch.index_select(vp.symbol("x2"), -1, indices) / ls,
        ])

    if expr.op == vtorch.primitives.prod_p:
        # pytorch's prod causes GPU synchronizes in backward pass
        expr = vctorch.fast_prod_positive(expr.args[0], **expr.kwargs)

    return expr


def to_visual(expr):
    if expr.op == vtorch.primitives.sum_p:
        if isinstance(expr.args[0], vp.Vexpr) \
           and expr.args[0].op == vctorch.primitives.mul_along_dim_p:
            w, t = expr.args[0].args
            assert t.op == vtorch.primitives.stack_p
            assert (not isinstance(t.args[0], vp.Vexpr)
                    and isinstance(t.args[0], (list, tuple)))
            # zip w with the stack
            expr = expr.update_args(
                tuple(w[i] * sum_operand
                      for i, sum_operand in enumerate(t.args[0]))
            )

    if expr.op == vctorch.primitives.mul_along_dim_p:
        # Detect any mul_along_dim that isn't used for a weighted sum.
        w, t = expr.args
        if t.op != vtorch.primitives.stack_p:
            expr = w * t

    if expr.op == vtorch.primitives.cdist_p:
        assert expr.args[0].op == select_divide_p
        names, ls = expr.args[0].args
        assert ls.op == vtorch.primitives.index_select_p
        symbol = ls.args[0]
        indices = ls.args[2]
        expr = expr.update_args(
            tuple(select(name) / symbol[index]
                  for name, index in zip(names, indices))
        )

    if expr.op in (vtorch.primitives.sum_p, vtorch.primitives.prod_p):
        if expr.kwargs.get("dim", None) == -3:
            # Convert to dim=0 and remove any child stack.
            if isinstance(expr.args[0], vp.Vexpr) \
               and expr.args[0].op == vtorch.primitives.stack_p:
                expr = vp.Vexpr(expr.op, expr.args[0].args, dict(dim=0))
            else:
                expr = vp.Vexpr(expr.op, expr.args, dict(dim=0))

    return expr


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
        ls = vtorch.index_select(lengthscale, -1, ls_indices)
        return ovt.matern(vtorch.cdist(select_divide(names, ls), p=2),
                          nu=2.5)

    def choice_kernel(names):
        ls_indices = ialloc.allocate(len(names))
        ls = vtorch.index_select(lengthscale, -1, ls_indices)
        return vtorch.exp(-vtorch.cdist(select_divide(names, ls),
                                        p=1))

    def scalar_factorized_and_joint(names, suffix):
        w_additive = vp.symbol("w_additive" + suffix)
        alpha_factorized_or_joint = vp.symbol("alpha_factorized_or_joint"
                                              + suffix)
        state.allocate(w_additive, (len(names),),
                       zero_one_exclusive(),
                       ol.priors.DirichletPrior(torch.full((len(names),), 2.0)))
        state.allocate(alpha_factorized_or_joint, (1,),
                       zero_one_exclusive(),
                       ol.priors.BetaPrior(4.0, 1.0))

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

    regime_kernel = vtorch.prod(
        vtorch.stack([scalar_kernel(regime_joint_names)]
                     + regime_kernels("_factorized"),
                     dim=-3),
        dim=-3)
    architecture_kernel = vtorch.prod(
        vtorch.stack([scalar_kernel(architecture_joint_names)]
                     + architecture_kernels("_factorized"),
                     dim=-3),
        dim=-3)
    joint_kernel = vtorch.prod(
        vtorch.stack([scalar_kernel(regime_joint_names
                                    + architecture_joint_names)]
                     + regime_kernels("_joint")
                     + architecture_kernels("_joint"),
                     dim=-3),
        dim=-3)

    alpha_regime_vs_architecture = vp.symbol("alpha_regime_vs_architecture")
    alpha_factorized_vs_joint = vp.symbol("alpha_factorized_vs_joint")
    scale = vp.symbol("scale")

    state.allocate(alpha_regime_vs_architecture, (1,),
                   zero_one_exclusive(),
                   ol.priors.BetaPrior(2.0, 2.0))
    state.allocate(alpha_factorized_vs_joint, (1,),
                   zero_one_exclusive(),
                   ol.priors.BetaPrior(4.0, 1.0))
    state.allocate(scale, (),
                   gpytorch.constraints.GreaterThan(1e-4),
                   gpytorch.priors.GammaPrior(2.0, 0.15))

    kernel = vctorch.mul_along_dim(
        scale,
        vtorch.sum(
            vctorch.mul_along_dim(
                vctorch.heads_tails(alpha_factorized_vs_joint),
                vtorch.stack(
                    [vtorch.sum(
                        vctorch.mul_along_dim(
                            vctorch.heads_tails(
                                alpha_regime_vs_architecture),
                            vtorch.stack([
                                regime_kernel,
                                architecture_kernel
                            ], dim=-3),
                            dim=-3),
                        dim=-3),
                     joint_kernel], dim=-3),
                dim=-3),
            dim=-3),
        dim=-3)

    state.allocate(lengthscale, (ialloc.count,),
                   gpytorch.constraints.GreaterThan(1e-4),
                   gpytorch.priors.GammaPrior(3.0, 6.0))

    kernel_runnable = vp.bottom_up_transform(partial(to_runnable, index_for_name),
                                    kernel)
    kernel_visualizable = vp.bottom_up_transform(to_visual, kernel)

    return kernel_runnable, state


# Lets long-running processes that instantiate multiple GPs avoid recompiling.
# Sadly, compiling is still required once for every batch_shape, since
# torch.vmap doesn't work with torch.compile, so each Vexpr needs to handle
# batching internally thus is different for each batch_shape. This caching will
# become much more powerful if we can switch to vmapping a torch.compiled
# function that is batch-unaware.
cached_compiled_fs = {}

class VexprKernel(gpytorch.kernels.Kernel):
    def __init__(self, kernel_vexpr, state_modules, batch_shape, vectorize=True,
                 torch_compile=False, initialize="mean"):
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

        if self.torch_compile:
            kernel_f2 = vp.to_python(self.kernel_vexpr)
        else:
            kernel_f2 = partial(vp.eval, self.kernel_vexpr)

        def kernel_f(x1, x2, parameters):
            return kernel_f2({"x1": x1, "x2": x2, **parameters})

        if self.torch_compile:
            comparable_vexpr = vp.comparable_hashable(self.kernel_vexpr)
            compiled_f = cached_compiled_fs.get(comparable_vexpr, None)
            if compiled_f is None:
                compiled_f = torch.compile(kernel_f)
                cached_compiled_fs[comparable_vexpr] = compiled_f
            kernel_f = compiled_f

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

        kernel_vexpr, state_builder = make_handson_kernel(xform.space2)
        state_modules = state_builder.instantiate(aug_batch_shape)
        covar_module = VexprKernel(
            kernel_vexpr,
            state_modules,
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
