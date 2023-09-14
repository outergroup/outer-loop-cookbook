from functools import partial

import botorch
import gpytorch
import outerloop as ol
import outerloop.vexpr.torch as ovt
import torch
import vexpr as vp
import vexpr.torch as vtorch


class ValueModule(gpytorch.Module):
    def __init__(self, shape, constraint, prior=None):
        super().__init__()
        name = "raw_value"
        self.register_parameter(
            name=name,
            parameter=torch.nn.Parameter(torch.zeros(*shape))
        )
        self.register_constraint(name, constraint)
        if prior is not None:
            self.register_prior(f"prior", prior,
                                ValueModule.get_value,
                                ValueModule.set_value)

    @staticmethod
    def get_value(instance):
        return instance.value

    @staticmethod
    def set_value(instance, v):
        if not torch.is_tensor(v):
            v = torch.as_tensor(v).to(instance.raw_value)

        instance.initialize(
            **{"raw_value":
               instance.raw_value_constraint.inverse_transform(v)}
        )

    @property
    def value(self):
        with torch.profiler.record_function("ValueModule.value"):
            if self.training:
                return self.raw_value_constraint.transform(self.raw_value)
            else:
                with torch.no_grad():
                    return self.raw_value_constraint.transform(self.raw_value)


class State:
    def __init__(self, batch_shape):
        self.batch_shape = batch_shape
        self.modules = {}

    def allocate(self, symbol, shape, constraint=None, prior=None):
        shape = self.batch_shape + shape
        name = symbol.args[0]
        self.modules[name] = ValueModule(shape, constraint, prior)


class IndexAllocator:
    def __init__(self):
        self.count = 0

    def allocate(self, n):
        ret = torch.arange(self.count, self.count + n)
        self.count += n
        return ret


N_HOT_PREFIX = "choice_nhot"


def make_handson_kernel(space, batch_shape=()):
    """
    This kernel attempts to group parameters into orthogonal groups, while also
    always allowing for the model to learn to use the joint space.
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
        return ovt.matern(vtorch.cdist(x1[..., indices] / lengthscale[ls_indices],
                                       x2[..., indices] / lengthscale[ls_indices],
                                   p=2),
                      nu=2.5)

    def choice_kernel(names):
        ls_indices = ialloc.allocate(len(names))
        indices = [index_for_name(name) for name in names]
        return ovt.matern(vtorch.cdist(x1[..., indices] / lengthscale[ls_indices],
                                       x2[..., indices] / lengthscale[ls_indices],
                                       p=1),
                      nu=2.5)


    def scalar_factorized_and_joint(names, suffix):
        w_additive = vp.symbol("w_additive" + suffix)
        w_factorized_or_joint = vp.symbol("w_factorized_or_joint" + suffix)
        state.allocate(w_additive, (len(names), 1, 1),
                       zero_one_exclusive(),
                       ol.priors.DirichletPrior(torch.full((len(names),), 2.0)))
        state.allocate(w_factorized_or_joint, (2, 1, 1),
                       zero_one_exclusive(),
                       ol.priors.BetaPrior(0.5, 0.5))
        return vtorch.sum(
            w_factorized_or_joint
            * vtorch.stack([
                vtorch.sum(
                    w_additive
                    * vtorch.stack([scalar_kernel([name])
                                    for name in names]),
                    dim=0),
                scalar_kernel(names),
            ]),
            dim=0
        )

    def regime_kernels():
        return [
            # kernel: regime choice parameters
            choice_kernel([f"{N_HOT_PREFIX}{i}"
                           for i in range(4)]),

            # kernel: lr schedule
            scalar_factorized_and_joint(["log_1cycle_initial_lr", "log_1cycle_final_lr",
                                         "log_1cycle_max_lr", "1cycle_pct_warmup"],
                                        "_lr"),

            # kernel: momentum schedule
            scalar_factorized_and_joint(["1cycle_max_momentum", "1cycle_min_momentum"],
                                        "_momentum"),

            # kernel: relative weight decay
            scalar_factorized_and_joint(["log_conv1_wd_div_gmean", "log_conv2_wd_div_gmean",
                                         "log_conv3_wd_div_gmean", "log_dense1_wd_div_gmean",
                                         "log_dense2_wd_div_gmean"],
                                        "_wd"),
        ]

    regime_joint_names = ["log_epochs", "log_batch_size", "log_gmean_weight_decay"]

    def architecture_kernels():
        return [
            # kernel: lr schedule
            scalar_factorized_and_joint(["log_conv1_channels_div_gmean",
                                         "log_conv2_channels_div_gmean",
                                         "log_conv3_channels_div_gmean",
                                         "log_dense1_units_div_gmean"],
                                        "_units_channels"),
        ]

    architecture_joint_names = ["log_gmean_channels_and_units"]

    regime_kernel = vtorch.prod(
        ([scalar_kernel(regime_joint_names)]
         + regime_kernels()),
        dim=0)
    architecture_kernel = vtorch.prod(
        ([scalar_kernel(architecture_joint_names)]
         + architecture_kernels()),
        dim=0)
    joint_kernel = vtorch.prod(
        ([scalar_kernel(regime_joint_names + architecture_joint_names)]
         + regime_kernels()
         + architecture_kernels()),
        dim=0)

    w_regime_vs_architecture = vp.symbol("w_regime_vs_architecture")
    w_factorized_vs_joint = vp.symbol("w_regime_vs_architecture")
    scale = vp.symbol("scale")

    state.allocate(w_regime_vs_architecture, (2, 1, 1),
                   zero_one_exclusive(),
                   ol.priors.BetaPrior(1.0, 1.0))
    state.allocate(w_factorized_vs_joint, (2, 1, 1),
                   zero_one_exclusive(),
                   ol.priors.BetaPrior(4.0, 1.0))
    state.allocate(scale, (),
                   gpytorch.constraints.GreaterThan(1e-4),
                   gpytorch.priors.GammaPrior(2.0, 0.15))

    kernel = (scale
              * vtorch.sum(w_factorized_vs_joint
                           * vtorch.stack([
                               vtorch.sum(w_regime_vs_architecture
                                          * vtorch.stack([
                                              regime_kernel,
                                              architecture_kernel
                                          ]),
                                          dim=0),
                               joint_kernel]),
                           dim=0))

    state.allocate(lengthscale, (ialloc.count,),
                   gpytorch.constraints.GreaterThan(1e-4),
                   gpytorch.priors.GammaPrior(3.0, 6.0))

    return kernel, state.modules


class VexprKernel(gpytorch.kernels.Kernel):
    def __init__(self, kernel_vexpr, state_modules, batch_shape):
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

    # def _apply(self, fn):
    #     self = super()._apply(self, fn)
    #     # TODO: this causes bugs by converting primitives into new instances
    #     # of primitives, breaking "is symbol_p" checks
    #     self.kernel_vexpr = tree_map(
    #         lambda v: (fn(v)
    #                    if isinstance(v, torch.Tensor)
    #                    else v),
    #         self.kernel_vexpr)
    #     return self

    def forward(self, x1, x2, diag: bool = False, last_dim_is_batch: bool = False):
        assert not diag
        assert not last_dim_is_batch

        parameters = {name: module.value
                      for name, module in self.state.items()}

        return self.kernel_f(x1, x2, parameters)


class VexprHandsOnLossModel(botorch.models.SingleTaskGP):
    def __init__(self, train_X, train_Y,
                 search_space,
                 search_xform,
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
            ol.transforms.multiply(
                {"1cycle_min_momentum_pct": "1cycle_min_momentum"},
                "1cycle_max_momentum"),
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
