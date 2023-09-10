from functools import partial

import gpytorch
import outerloop as ol
import outerloop.vexpr.torch as ovt
import torch
import vexpr as vp
import vexpr.torch as vtorch


class State:
    def __init__(self, batch_shape):
        raise NotImplementedError()

    def allocate(self, symbol, shape, constraint=None, prior=None):
        pass

    def get_symbols(self):
        pass


class IndexAllocator:
    def __init__(self):
        self.count = 0

    def allocate(self, n):
        ret = torch.arange(self.count, self.count + n)
        self.count += n
        return ret


def make_handson_kernel(space):
    """
    This kernel groups parameters into orthogonal groups.
    """
    zero_one_exclusive = partial(gpytorch.constraints.Interval,
                                 1e-6,
                                 1 - 1e-6)

    state = State()

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
        state.allocate(w_additive, len(names),
                       zero_one_exclusive(),
                       ol.priors.DirichletPrior(torch.full(len(names), 2.0)))
        state.allocate(w_factorized_or_joint, 2,
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
            choice_kernel(["regime_choice_nhot0", "regime_choice_nhot1",
                           "regime_choice_nhot2", "regime_choice_nhot3"]),

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

    regime_kernel = torch.prod(
        ([scalar_kernel(regime_joint_names)]
         + regime_kernels()),
        dim=0)
    architecture_kernel = torch.prod(
        ([scalar_kernel(architecture_joint_names)]
         + architecture_kernels()),
        dim=0)
    joint_kernel = torch.prod(
        ([scalar_kernel(regime_joint_names + architecture_joint_names)]
         + regime_kernels()
         + architecture_kernels()),
        dim=0)

    w_regime_vs_architecture = vp.symbol("w_regime_vs_architecture")
    w_factorized_vs_joint = vp.symbol("w_regime_vs_architecture")
    scale = vp.symbol("scale")

    state.allocate(w_regime_vs_architecture, (2,),
                   zero_one_exclusive(),
                   ol.priors.BetaPrior(1.0, 1.0))
    state.allocate(w_factorized_vs_joint, (2,),
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
                               joint_kernel])))

    state.allocate(lengthscale, ialloc.count,
                   gpytorch.constraints.GreaterThan(1e-4),
                   gpytorch.priors.GammaPrior(3.0, 6.0))

    return kernel, state
