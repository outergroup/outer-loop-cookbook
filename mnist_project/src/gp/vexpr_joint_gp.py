from functools import partial

import botorch
import gpytorch
import outerloop as ol
import outerloop.vexpr.torch as ovt
import torch
import vexpr as vp
import vexpr.torch as vtorch
import vexpr.custom.torch as vctorch

from .gp_utils import StateBuilder, IndexAllocator, FastStandardize, VexprKernel


N_HOT_PREFIX = "choice_nhot"


def make_botorch_range_choice_kernel(space):
    """
    Creates a kernel vexpr and an object that is ready to instantiate kernel
    parameters, given batch shape.

    This kernel is similar to Botorch's MixedSingleTaskGP. The two are
    equivalent, just with different parameterizations and priors. The
    multiplicative weights are parameterized differently, and this nhot kernel
    uses the sum, not the mean (this is equivalent to having a different
    lengthscale priors).
    """
    zero_one_exclusive = partial(gpytorch.constraints.Interval,
                                 1e-6,
                                 1 - 1e-6)

    state = StateBuilder()

    ialloc = IndexAllocator()

    lengthscale = vp.symbol("lengthscale")
    x1 = vp.symbol("x1")
    x2 = vp.symbol("x2")

    range_indices = []
    nhot_indices = []
    for i, p in enumerate(space):
        if N_HOT_PREFIX in p.name:
            nhot_indices.append(i)
        else:
            range_indices.append(i)
    range_indices = torch.tensor(range_indices)
    nhot_indices = torch.tensor(nhot_indices)

    def range_kernel():
        ls_indices = ialloc.allocate(len(range_indices))
        ls = vtorch.unsqueeze(vtorch.index_select(lengthscale, -1, ls_indices),
                              -2)
        return ovt.matern(
            vtorch.cdist(
                vtorch.index_select(x1, -1, range_indices)
                / ls,
                vtorch.index_select(x2, -1, range_indices)
                / ls,
                p=2),
            nu=2.5)

    def nhot_kernel():
        ls_indices = ialloc.allocate(len(nhot_indices))
        ls = vtorch.unsqueeze(vtorch.index_select(lengthscale, -1, ls_indices),
                              -2)
        return vtorch.exp(
            -vtorch.cdist(
                vtorch.index_select(x1, -1, nhot_indices)
                / ls,
                vtorch.index_select(x2, -1, nhot_indices)
                / ls,
                p=1)
        )

    alpha_range_vs_nhot = vp.symbol("alpha_range_vs_nhot")
    state.allocate(alpha_range_vs_nhot, (1,),
                   zero_one_exclusive(),
                   ol.priors.BetaPrior(2.0, 2.0))
    sum_kernel = vtorch.sum(
        vctorch.mul_along_dim(
            vctorch.heads_tails(alpha_range_vs_nhot),
            vtorch.stack([range_kernel(), nhot_kernel()], dim=-3),
            dim=-3),
        dim=-3)
    prod_kernel = vctorch.fast_prod_positive(
        vtorch.stack([range_kernel(), nhot_kernel()],
                     dim=-3),
        dim=-3)

    alpha_factorized_vs_joint = vp.symbol("alpha_factorized_vs_joint")
    state.allocate(alpha_factorized_vs_joint, (1,),
                   zero_one_exclusive(),
                   ol.priors.BetaPrior(2.0, 2.0))
    scale = vp.symbol("scale")
    state.allocate(scale, (),
                   gpytorch.constraints.GreaterThan(1e-4),
                   gpytorch.priors.GammaPrior(2.0, 0.15))
    kernel = vctorch.mul_along_dim(
        scale,
        vtorch.sum(
            vctorch.mul_along_dim(
                vctorch.heads_tails(alpha_factorized_vs_joint),
                vtorch.stack([sum_kernel, prod_kernel], dim=-3),
                dim=-3),
            dim=-3),
        dim=-3)

    state.allocate(lengthscale, (ialloc.count,),
                   gpytorch.constraints.GreaterThan(1e-4),
                   gpytorch.priors.GammaPrior(3.0, 6.0))

    return kernel, state


class VexprFullyJointGP(botorch.models.SingleTaskGP):
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
            partial(ol.transforms.ChoiceNHotProjection, out_name=N_HOT_PREFIX)
        ]

        xform = ol.transforms.Chain(search_space, *xforms)

        kernel_vexpr, state_builder = make_botorch_range_choice_kernel(
            xform.space2)
        state_modules = state_builder.instantiate(aug_batch_shape)
        covar_module = VexprKernel(kernel_vexpr, state_modules,
                                   batch_shape=aug_batch_shape,
                                   vectorize=vectorize,
                                   torch_compile=torch_compile)

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
            noise_prior=gpytorch.priors.GammaPrior(1.1, 0.05),
            # noise_prior=gpytorch.priors.GammaPrior(0.9, 10.0),
        )

        super().__init__(
            train_X, train_Y,
            input_transform=input_transform,
            covar_module=covar_module,
            likelihood=likelihood,
            **extra_kwargs
        )
