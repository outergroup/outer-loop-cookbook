import glob
import os

import botorch
import gpytorch
import torch
import outerloop as ol

from ..gp.gp_utils import configs_dirs_to_X_Y


SMOKE_TEST = os.environ.get("SMOKE_TEST")
NUM_RESTARTS = 200 if not SMOKE_TEST else 2
RAW_SAMPLES = 2048 if not SMOKE_TEST else 32
MC_SAMPLES = 512 if not SMOKE_TEST else 32


def perform_bo(model, X, Y, X_pending, bounds):
    objective = botorch.acquisition.GenericMCObjective(lambda Z: Z[..., 0])
    # Numerically approximate the expected best value among the current set of
    # observations.
    best_f = objective(
        botorch.sampling.SobolQMCNormalSampler(
            sample_shape=torch.Size([MC_SAMPLES]))(
                model.posterior(X)
            )
        ).max(dim=-1).values.mean()

    # perform BO
    candidates, acq_value = botorch.optim.optimize_acqf(
        acq_function=botorch.acquisition.qExpectedImprovement(
            model=model,
            best_f=best_f,
            sampler=botorch.sampling.SobolQMCNormalSampler(
                sample_shape=torch.Size([MC_SAMPLES])),
            objective=objective,
            X_pending=X_pending,
        ),
        bounds=bounds,
        q=1,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
    )
    print("acq_value", acq_value.item())

    return candidates.detach()


class BO:
    def __init__(self, model_cls, metric, space, xform):
        self.model_cls = model_cls
        self.metric = metric
        self.parameter_space = space
        self.xform = xform
        if xform is not None:
            self.search_space = xform.space2
            self.rounding_function = ol.transforms.UntransformThenTransform(
                self.search_space, self.xform
            ).transform
        else:
            self.search_space = space
            self.rounding_function = None

    def __call__(self, trial_dir, prev_configs, prev_trial_dirs, pending_configs):
        # TODO find better place for this
        torch.set_default_dtype(torch.float64)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        X, Y = configs_dirs_to_X_Y(prev_configs, prev_trial_dirs, self.metric,
                                   self.parameter_space, self.xform, device)
        model = self.model_cls(X, Y).to(device)
        model = cached_fit_gp(model, prev_trial_dirs, trial_dir)
        
        if len(pending_configs) > 0:
            X_pending = ol.configs_to_X(self.parameter_space, pending_configs,
                                        xform=self.xform.transform, device=device)
        else:
            X_pending = None

        bounds = ol.botorch_bounds(self.search_space).to(device)
        objective = botorch.acquisition.GenericMCObjective(lambda Z: Z[..., 0])
        # Numerically approximate the expected best value among the current set of
        # observations.
        best_f = objective(
            botorch.sampling.SobolQMCNormalSampler(
                sample_shape=torch.Size([MC_SAMPLES]))(
                    model.posterior(X)
                )
            ).max(dim=-1).values.mean()

        has_choice_params = any(isinstance(p, ol.Choice) for p in self.search_space)
        if has_choice_params:
            base_configs = ol.all_base_configurations(self.search_space)
            candidates, acq_value = botorch.optim.optimize_acqf_mixed(
                acq_function=botorch.acquisition.qExpectedImprovement(
                    model=model,
                    best_f=best_f,
                    sampler=botorch.sampling.SobolQMCNormalSampler(
                        sample_shape=torch.Size([MC_SAMPLES])),
                    objective=objective,
                ),
                bounds=bounds,
                fixed_features_list=base_configs,
                q=1,
                num_restarts=NUM_RESTARTS,
                raw_samples=RAW_SAMPLES,
                post_processing_func=self.rounding_function,
            )
        else:
            candidates, acq_value = botorch.optim.optimize_acqf(
                acq_function=botorch.acquisition.qExpectedImprovement(
                    model=model,
                    best_f=best_f,
                    sampler=botorch.sampling.SobolQMCNormalSampler(
                        sample_shape=torch.Size([MC_SAMPLES])),
                    objective=objective,
                    X_pending=X_pending,
                ),
                bounds=bounds,
                q=1,
                num_restarts=NUM_RESTARTS,
                raw_samples=RAW_SAMPLES,
                post_processing_func=self.rounding_function,
            )

        print("acq_value", acq_value.item())
        X_new = candidates.detach()

        prev_configs = ol.X_to_configs(self.parameter_space, X_new,
                                  self.xform.untransform)
        assert len(prev_configs) == 1
        args = prev_configs[0]
        return args


def get_latest_state_dict(prev_trial_dirs):
    highest_version = None
    highest_version_path = None
    for trial_dir in prev_trial_dirs:
        gp_dict_paths = glob.glob(os.path.join(trial_dir, "gp_dict_*.pt"))
        assert 0 <= len(gp_dict_paths) <= 1
        if len(gp_dict_paths) > 0:
            version = int(gp_dict_paths[0].split("_")[-1].split(".")[0])
            if highest_version is None or version > highest_version:
                highest_version = version
                highest_version_path = gp_dict_paths[0]

    prev_gp_dict_version = highest_version
    if highest_version is not None:
        gp_dict = torch.load(highest_version_path)
    else:
        gp_dict = None

    return gp_dict, highest_version


def cached_fit_gp(model, prev_trial_dirs, trial_dir=None, **kwargs):
    gp_dict, prev_gp_dict_version = get_latest_state_dict(prev_trial_dirs)
    new_gp_dict_version = len(model.train_inputs[0])

    if gp_dict is not None:
        # gpytorch includes prior constants and bound constants in the state
        # dict, which was a bad decision on their part
        gp_dict = {k: v for k, v in gp_dict.items()
                   if "prior" not in k and "bound" not in k}
        model.load_state_dict(gp_dict, strict=False)

    if prev_gp_dict_version is None \
    or prev_gp_dict_version < new_gp_dict_version:
        botorch.fit_gpytorch_mll(
            gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
        )
        if trial_dir is not None:
            gp_dict_path = os.path.join(trial_dir, f"gp_dict_{new_gp_dict_version}.pt")
            torch.save(model.state_dict(), gp_dict_path)

    return model
