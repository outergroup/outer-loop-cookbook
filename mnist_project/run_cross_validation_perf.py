import os
import time
import shutil
import pprint
from functools import partial

import botorch
import torch
import gpytorch
from botorch.cross_validation import batch_cross_validation, gen_loo_cv_folds

from src.sweeps import CONFIGS
from src import gen
from src.gp import mnist_metrics, MODELS

from src.sweeps import mnist1
from src.gp.gp_utils import configs_dirs_to_X_Y
from src.gp.mnist_metrics import trial_dir_to_loss_y
from src.scheduling import parse_results

SMOKE_TEST = os.environ.get("SMOKE_TEST")


def run(sweep_name, model_name, trace=False):
    # import logging
    # logging.basicConfig()
    # logging.getLogger().setLevel(logging.DEBUG)
    torch.set_default_dtype(torch.float64)

    # Toggle for perf
    debug = False
    gpytorch.settings.debug._set_state(debug)
    botorch.settings.debug._set_state(debug)

    config = CONFIGS[sweep_name]
    model_cls = MODELS[model_name]

    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)
    sweep_dir = os.path.join(project_dir, "results", sweep_name)

    search_space = config["search_space"]
    search_xform = config["search_xform"].to(device)

    model_cls = partial(model_cls,
                        search_space=search_space,
                        search_xform=search_xform,
                        round_inputs=False)

    configs, trial_dirs, _ = parse_results(sweep_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, Y = configs_dirs_to_X_Y(configs, trial_dirs, trial_dir_to_loss_y,
                               search_space, search_xform, device=device)

    n_cv = 5
    cv_folds = gen_loo_cv_folds(train_X=X[:n_cv], train_Y=Y[:n_cv])

    if trace:
        from torch.profiler import profile, record_function, ProfilerActivity
        group_by_shape = False
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=group_by_shape) as prof:
            with record_function("optimization_test"):
                cv_results = batch_cross_validation(
                    model_cls=model_cls,
                    mll_cls=gpytorch.mlls.ExactMarginalLogLikelihood,
                    cv_folds=cv_folds,
                    observation_noise=True,
                )
        print(prof.key_averages(group_by_input_shape=group_by_shape).table(sort_by="cuda_time_total",
                                                                           row_limit=20))
        filename = f"cross_validate-{model_name}.json"
        prof.export_chrome_trace(filename)
        print("Saved", filename)
    else:
        tstart = time.time()
        cv_results = batch_cross_validation(
            model_cls=model_cls,
            mll_cls=gpytorch.mlls.ExactMarginalLogLikelihood,
            cv_folds=cv_folds,
        )
        tend = time.time()
        print(f"Elapsed time: {tend - tstart:>2f}")



def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="VexprHandsOnLossModel")
    parser.add_argument("--sweep-name", type=str, default="mnist1")
    parser.add_argument("--trace", action="store_true")

    cmd_args = parser.parse_args()
    run(**vars(cmd_args))


if __name__ == "__main__":
    main()
