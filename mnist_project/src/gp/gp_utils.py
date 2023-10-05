import botorch
import gpytorch
import linear_operator
import outerloop as ol
import torch


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


class State:
    def __init__(self, batch_shape):
        self.batch_shape = batch_shape
        self.modules = {}

    def allocate(self, symbol, shape, constraint=None, prior=None, initialize=None):
        shape = self.batch_shape + shape
        name = symbol.args[0]
        assert name not in self.modules
        self.modules[name] = ValueModule(shape, constraint, prior, initialize)


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


