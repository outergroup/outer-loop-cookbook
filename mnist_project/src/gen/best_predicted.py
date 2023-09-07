import os

import botorch
import gpytorch
import torch

from ..gp.gp_utils import configs_dirs_to_X_Y
from ..scheduling import parse_results


class BestPredicted:
    def __init__(self, source_sweep_name, model_cls, metric, space, xform):
        self.source_sweep_name = source_sweep_name
        self.model_cls = model_cls
        self.metric = metric
        self.parameter_space = space
        self.xform = xform
        if xform is not None:
            self.search_space = xform.space2
        else:
            self.search_space = space

    def __call__(self, trial_dir, prev_configs, prev_trial_dirs, pending_configs):
        # TODO find better place for this
        torch.set_default_dtype(torch.float64)

        (prev_configs,
         prev_trial_dirs,
         _) = parse_results(self.source_sweep_name)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        X, Y = configs_dirs_to_X_Y(prev_configs, prev_trial_dirs,
                                   self.metric, self.parameter_space,
                                   self.xform, device=device)
        model = self.model_cls(X, Y).to(device)

        botorch.fit_gpytorch_mll(
            gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
        )

        with torch.no_grad():
            Y_pred = model.posterior(X).mean

        i_best = Y_pred.unsqueeze(-1).argmax().item()
        best_config = prev_configs[i_best]

        print(best_config)
        return best_config
