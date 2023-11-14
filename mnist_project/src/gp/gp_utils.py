from functools import partial

import botorch
import gpytorch
import linear_operator
import outerloop as ol
import outerloop.vexpr.torch as ovt
import torch
import vexpr as vp
import vexpr.core
import vexpr.torch as vtorch
import vexpr.custom.torch as vctorch
from vexpr.core import _p_and_constructor


def configs_dirs_to_X_Y(configs, trial_dirs, metric, space, xform,
                        device=None):
    kept_args = []
    kept_y = []
    for args, trial_dir in zip(configs, trial_dirs):
        y = metric(trial_dir)
        if y is not None:
            kept_args.append(args)
            kept_y.append(y)

    X = ol.configs_to_X(space, kept_args,
                        xform=xform.transform, device=device)
    Y = torch.tensor(kept_y, device=device)

    return X, Y


class ValueModule(gpytorch.Module):
    def __init__(self, shape, constraint, prior=None, initialize=None):
        super().__init__()
        name = "raw_value"
        self.register_parameter(
            name=name,
            parameter=torch.nn.Parameter(torch.zeros(shape))
        )
        self.register_constraint(name, constraint)
        if prior is not None:
            self.register_prior(f"prior", prior,
                                ValueModule.botorch_get_value,
                                ValueModule.set_value)

            if initialize is not None:
                if initialize == "mode":
                    value = prior.mode
                elif initialize == "mean":
                    value = prior.mean
                else:
                    raise ValueError(f"Unrecognized initialization: {initialize}")

                ValueModule.set_value(self, value)

    @staticmethod
    def botorch_get_value(instance):
        # botorch "sample_all_priors" expects an extra dimension at the end
        return instance.value.unsqueeze(-1)

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


class StateBuilder:
    def __init__(self):
        self.args_by_module_name = {}
        self.modules = {}

    def allocate(self, symbol, shape, constraint=None, prior=None, initialize=None):
        name = symbol.args[0]
        assert name not in self.args_by_module_name
        self.args_by_module_name[name] = shape, constraint, prior, initialize

    def instantiate(self, batch_shape):
        return {name: ValueModule(batch_shape + shape, constraint, prior,
                                  initialize)
                for name, (shape, constraint, prior, initialize)
                in self.args_by_module_name.items()}


class IndexAllocator:
    def __init__(self):
        self.count = 0

    def allocate(self, n):
        ret = torch.arange(self.count, self.count + n)
        self.count += n
        return ret



class FastStandardize(botorch.models.transforms.Standardize):
    def untransform(self, Y, Yvar=None):
        """Override to avoid a CUDA synchronize, see *CHANGE* below."""

        # *CHANGE* Remove this check, because self._is_trained is a bool tensor
        # stored on the device, so this forces a CUDA synchronize
        #
        # if not self._is_trained:
        #     raise RuntimeError(
        #         "`Standardize` transforms must be called on outcome data "
        #         "(e.g. `transform(Y)`) before calling `untransform`, since "
        #         "means and standard deviations need to be computed."
        #     )
        Y_utf = self.means + self.stdvs * Y
        Yvar_utf = self._stdvs_sq * Yvar if Yvar is not None else None
        return Y_utf, Yvar_utf


    def untransform_posterior(self, posterior):
        """Override to avoid a CUDA synchronize, see *CHANGE* below."""
        if self._outputs is not None:
            raise NotImplementedError(
                "Standardize does not yet support output selection for "
                "untransform_posterior"
            )
        # *CHANGE* Remove this check, because self._is_trained is a bool tensor
        # stored on the device, so this forces a CUDA synchronize
        # if not self._is_trained:
        #     raise RuntimeError(
        #         "`Standardize` transforms must be called on outcome data "
        #         "(e.g. `transform(Y)`) before calling `untransform_posterior`, since "
        #         "means and standard deviations need to be computed."
        #     )
        is_mtgp_posterior = False
        if type(posterior) is botorch.posteriors.GPyTorchPosterior:
            is_mtgp_posterior = posterior._is_mt
        if not self._m == posterior._extended_shape()[-1] and not is_mtgp_posterior:
            raise RuntimeError(
                "Incompatible output dimensions encountered. Transform has output "
                f"dimension {self._m} and posterior has "
                f"{posterior._extended_shape()[-1]}."
            )

        if type(posterior) is not botorch.posteriors.GPyTorchPosterior:
            # fall back to TransformedPosterior
            # this applies to subclasses of GPyTorchPosterior like MultitaskGPPosterior
            return botorch.posteriors.TransformedPosterior(
                posterior=posterior,
                sample_transform=lambda s: self.means + self.stdvs * s,
                mean_transform=lambda m, v: self.means + self.stdvs * m,
                variance_transform=lambda m, v: self._stdvs_sq * v,
            )
        # GPyTorchPosterior (TODO: Should we Lazy-evaluate the mean here as well?)
        mvn = posterior.distribution
        offset = self.means
        scale_fac = self.stdvs
        if not posterior._is_mt:
            mean_tf = offset.squeeze(-1) + scale_fac.squeeze(-1) * mvn.mean
            scale_fac = scale_fac.squeeze(-1).expand_as(mean_tf)
        else:
            mean_tf = offset + scale_fac * mvn.mean
            reps = mean_tf.shape[-2:].numel() // scale_fac.size(-1)
            scale_fac = scale_fac.squeeze(-2)
            if mvn._interleaved:
                scale_fac = scale_fac.repeat(*[1 for _ in scale_fac.shape[:-1]], reps)
            else:
                scale_fac = torch.repeat_interleave(scale_fac, reps, dim=-1)

        if (
            not mvn.islazy
            # TODO: Figure out attribute namming weirdness here
            or mvn._MultivariateNormal__unbroadcasted_scale_tril is not None
        ):
            # if already computed, we can save a lot of time using scale_tril
            covar_tf = linear_operator.operators.CholLinearOperator(mvn.scale_tril * scale_fac.unsqueeze(-1))
        else:
            lcv = mvn.lazy_covariance_matrix
            scale_fac = scale_fac.expand(lcv.shape[:-1])
            scale_mat = linear_operator.operators.DiagLinearOperator(scale_fac)
            covar_tf = scale_mat @ lcv @ scale_mat

        kwargs = {"interleaved": mvn._interleaved} if posterior._is_mt else {}
        mvn_tf = mvn.__class__(mean=mean_tf, covariance_matrix=covar_tf, **kwargs)
        return botorch.posteriors.GPyTorchPosterior(mvn_tf)


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
        self.kernel_vexpr = vp.transform_leafs(
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


# Vexpr primitives that never get run. Some get translated into runnable
# primitives, others are intended for visualization.
select_divide_p, select_divide = _p_and_constructor("select_divide")
sum_p, sum_ = _p_and_constructor("sum")
prod_p, prod_ = _p_and_constructor("prod")
exp_p, exp = _p_and_constructor("exp")
compare_p, compare = _p_and_constructor("compare")
matern_25_p, matern_25 = _p_and_constructor("matern_25")
norm_l1_p, norm_l1 = _p_and_constructor("norm_l1")
norm_l2_p, norm_l2 = _p_and_constructor("norm_l2")


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
            # zip w with the stack, moving comments up to above the
            # multiplication
            new_arg0 = [
                sum_operand.new(vp.primitives.operator_mul_p,
                                (w[i],
                                 vp.Vexpr(sum_operand.op,
                                          sum_operand.args,
                                          sum_operand.kwargs)),
                                {})
                for i, sum_operand in enumerate(t.args[0])]
            expr = expr.update_args((new_arg0,))

    if expr.op == vctorch.primitives.mul_along_dim_p:
        # Detect any mul_along_dim that isn't used for a weighted sum.
        w, t = expr.args
        if t.op != vtorch.primitives.stack_p:
            expr = w * t

    if expr.op == vtorch.primitives.cdist_p:
        assert expr.args[0].op == select_divide_p
        names, ls = expr.args[0].args
        if ls.op == vtorch.primitives.unsqueeze_p:
            ls = ls.args[0]

        assert ls.op == vtorch.primitives.index_select_p
        symbol = ls.args[0]
        indices = ls.args[2]
        new_arg0 = [compare(name) / symbol[index]
                    for name, index in zip(names, indices)]
        p = expr.kwargs.get("p", 2)
        if p == 1:
            expr = norm_l1(new_arg0)
        elif p == 2:
            expr = norm_l2(new_arg0)
        else:
            raise ValueError(p)

    if expr.op == ovt.primitives.matern_p:
        assert expr.kwargs.get("mu", 2.5) == 2.5
        expr = matern_25(*expr.args)

    if expr.op == vtorch.primitives.exp_p:
        expr = exp(*expr.args)

    if expr.op in (vtorch.primitives.sum_p, vtorch.primitives.prod_p):
        if expr.kwargs.get("dim", None) == -3:
            # Convert to dim=0 and remove any child stack.
            if isinstance(expr.args[0], vp.Vexpr) \
               and expr.args[0].op == vtorch.primitives.stack_p:
                expr = expr.new(expr.op, expr.args[0].args, dict(dim=0))
            else:
                expr = expr.new(expr.op, expr.args, dict(dim=0))

        if expr.kwargs.get("dim", None) == 0:
            op = sum_p if expr.op == vtorch.primitives.sum_p else prod_p
            expr = expr.new(op, expr.args, {})

    return expr
