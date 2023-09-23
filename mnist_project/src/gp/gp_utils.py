import gpytorch
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
    def __init__(self, shape, constraint, prior=None, initialize="mean"):
        super().__init__()
        name = "raw_value"
        self.register_parameter(
            name=name,
            parameter=torch.nn.Parameter(torch.zeros(*shape))
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
