import torch
import outerloop as ol


def unit_hypercube_to_search_space(space, X):
    X = X.clone()
    for i, p in enumerate(space):
        X[..., i] *= (p.bounds[1] - p.bounds[0])
        X[..., i] += p.bounds[0]
    return X


def mark_inactive_parameters(space, x):
    x = x.clone()

    choices = {p.name: p.to_user_space(x[i])
               for i, p in enumerate(space)
               if isinstance(p, ol.Choice)}
    for i, p in enumerate(space):
        if isinstance(p, ol.Choice):
            active = p.condition is None or p.condition(choices)
            if not active:
                del choices[p.name]
                x[i] = p.inactive_value
    for i, p in enumerate(space):
        if not isinstance(p, ol.Choice):
            active = p.condition is None or p.condition(choices)
            if not active:
                x[i] = p.inactive_value
    return x

class Sobol:
    def __init__(self, space, xform, seed=42):
        self.parameter_space = space
        self.xform = xform
        self.seed = seed
        if xform is not None:
            self.search_space = xform.space2

            self.rounding_function = ol.transforms.UntransformThenTransform(
                self.search_space, self.xform
            ).transform
                                  
        else:
            self.search_space = space
            self.rounding_function = None

    def __call__(self, trial_dir, prev_trials, pending_configs):
        # TODO find better place for this
        torch.set_default_dtype(torch.float64)

        prev_N_tot = len(prev_trials) + len(pending_configs)

        engine = torch.quasirandom.SobolEngine(
            dimension=len(self.search_space), scramble=True, seed=self.seed
        ).fast_forward(prev_N_tot)

        N = 1
        X = engine.draw(N, dtype=torch.double)
        X = unit_hypercube_to_search_space(self.search_space, X)

        if self.rounding_function is not None:
            X = self.rounding_function(X)
        X[0] = mark_inactive_parameters(self.search_space, X[0])

        configs = ol.X_to_configs(self.parameter_space, X, self.xform.untransform)

        args = configs[0]
        return args