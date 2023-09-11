import os
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


def run(sweep_name, model_name):
    # import logging
    # logging.basicConfig()
    # logging.getLogger().setLevel(logging.DEBUG)
    torch.set_default_dtype(torch.float64)

    # Toggle for perf
    debug = True
    gpytorch.settings.debug._set_state(debug)
    botorch.settings.debug._set_state(debug)

    config = CONFIGS[sweep_name]
    model_cls = MODELS[model_name]

    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)
    sweep_dir = os.path.join(project_dir, "results", sweep_name)

    model_cls = partial(model_cls,
                        search_space=mnist1.xform.space2,
                        search_xform=mnist1.xform,
                        round_inputs=False)


    if SMOKE_TEST:
        n_cvs = [10, 20]
    else:
        n_cvs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]

    configs, trial_dirs, _ = parse_results(sweep_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, Y = configs_dirs_to_X_Y(configs, trial_dirs, trial_dir_to_loss_y,
                               mnist1.space, mnist1.xform, device=device)

    for n_cv in n_cvs:
        print(f"# of examples: {n_cv}")
        cv_folds = gen_loo_cv_folds(train_X=X[:n_cv], train_Y=Y[:n_cv])

        cv_results = batch_cross_validation(
            model_cls=model_cls,
            mll_cls=gpytorch.mlls.ExactMarginalLogLikelihood,
            cv_folds=cv_folds,
            observation_noise=True,
        )


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="VexprHandsOnLossModel")
    parser.add_argument("--sweep-name", type=str, default="mnist1")

    cmd_args = parser.parse_args()
    run(**vars(cmd_args))


if __name__ == "__main__":
    main()
