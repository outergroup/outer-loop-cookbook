import copy
import os
import shutil
import pprint
import random
from functools import partial

import botorch
import torch
import gpytorch
from botorch.cross_validation import batch_cross_validation, gen_loo_cv_folds

from src.sweeps import CONFIGS
from src import gen
from src.gp import mnist_metrics, MODELS

from src.gp.gp_utils import configs_dirs_to_X_Y
from src.gp.mnist_metrics import trial_dir_to_loss_y
from src.scheduling import parse_results

SMOKE_TEST = os.environ.get("SMOKE_TEST")


def run(sweep_name, model_name, shuffle_seeds=None, vectorize=False,
        compile=False, trace=False):
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    search_space = config["search_space"]
    search_xform = config["search_xform"].to(device)

    model_cls = partial(model_cls,
                        search_space=search_space,
                        search_xform=search_xform,
                        round_inputs=False,
                        vectorize=vectorize,
                        torch_compile=compile)

    if SMOKE_TEST:
        n_cvs = [10, 20]
    else:
        n_cvs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]

    for shuffle_seed in shuffle_seeds:
        configs, trial_dirs, _ = parse_results(sweep_name)
        if shuffle_seed is not None:
            shuffled = list(zip(configs, trial_dirs))
            random.Random(shuffle_seed).shuffle(shuffled)
            configs, trial_dirs = zip(*shuffled)

        X, Y = configs_dirs_to_X_Y(configs, trial_dirs, trial_dir_to_loss_y,
                                   config["parameter_space"],
                                   search_xform,
                                   device=device)

        suffix = ""
        if shuffle_seed is not None:
            suffix = f"-seed{shuffle_seed}"

        result_dir = "results/cv"
        os.makedirs(result_dir, exist_ok=True)
        for n_cv in n_cvs:
            print(f"# of examples: {n_cv}")
            cv_folds = gen_loo_cv_folds(train_X=X[:n_cv], train_Y=Y[:n_cv])

            # optimization_log = []

            # def callback(parameters, result):
            #     optimization_log.append((copy.deepcopy(parameters), result))

            if trace:
                from torch.profiler import profile, record_function, ProfilerActivity
                group_by_shape = False
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                             record_shapes=group_by_shape) as prof:
                    with record_function("batch_cross_validation"):
                        cv_results = batch_cross_validation(
                            model_cls=model_cls,
                            mll_cls=gpytorch.mlls.ExactMarginalLogLikelihood,
                            cv_folds=cv_folds,
                            observation_noise=True,
                        )
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                print(prof.key_averages(group_by_input_shape=group_by_shape).table(
                    sort_by="cuda_time_total", row_limit=50))
                filename = f"cross_validation_{sweep_name}_{model_name}_{n_cv}.json"
                prof.export_chrome_trace(filename)
                print("Saved", filename)
            else:
                cv_results = batch_cross_validation(
                    model_cls=model_cls,
                    mll_cls=gpytorch.mlls.ExactMarginalLogLikelihood,
                    cv_folds=cv_folds,
                    observation_noise=True,
                    # fit_args=dict(
                    #     optimizer_kwargs=dict(
                    #         callback=callback
                    #     )
                    # )
                )

            filename = os.path.join(result_dir, f"cv-{sweep_name}-{model_name}{suffix}-{n_cv}.pt")
            result = {
                "state_dict": cv_results.model.state_dict(),
                "posterior": cv_results.posterior,
                "observed_Y": cv_results.observed_Y.cpu(),
                # "optimization_log": optimization_log,
            }
            print(f"Saving {filename}")
            torch.save(result, filename)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="VexprHandsOnGP")
    parser.add_argument("--sweep-name", type=str, default="mnist1")
    parser.add_argument("--vectorize", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--shuffle-seeds", type=int, default=[None], nargs="+")
    parser.add_argument("--trace", action="store_true")

    cmd_args = parser.parse_args()
    run(**vars(cmd_args))


if __name__ == "__main__":
    main()
