import os
import random
from functools import partial

import botorch
import torch
import gpytorch
from botorch.cross_validation import gen_loo_cv_folds, CVFolds, CVResults

from src.sweeps import CONFIGS
from src.gp import MODELS

from src.gp.gp_utils import configs_dirs_to_X_Y
from src.gp.mnist_metrics import trial_dir_to_loss_y
from src.scheduling import parse_results
from src.visuals import MeanNoiseKernelDistributionVisual

SMOKE_TEST = os.environ.get("SMOKE_TEST")


def run(sweep_name, model_name, shuffle_seeds=None, vectorize=False,
        compile=False, trace=False, visualize=True):
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
        n_cvs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]

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

        mll_cls = gpytorch.mlls.ExactMarginalLogLikelihood
        result_dir = "results/cv"
        os.makedirs(result_dir, exist_ok=True)
        for n_cv in n_cvs:
            print(f"# of examples: {n_cv}")
            cv_folds = gen_loo_cv_folds(train_X=X[:n_cv], train_Y=Y[:n_cv])

            def run_cross_validation():
                model_cv = model_cls(cv_folds.train_X, cv_folds.train_Y)
                mll_cv = mll_cls(model_cv.likelihood, model_cv)
                mll_cv.to(cv_folds.train_X)

                if visualize:
                    visual = MeanNoiseKernelDistributionVisual(model_cv, num_values_per_param=n_cv)

                    def callback(parameters, result):
                        """
                        Note: botorch will wrap this callback in slow code
                        """
                        visual.on_update(model_cv)

                    fit_args = dict(
                        optimizer_kwargs=dict(
                            callback=callback
                        )
                    )
                else:
                    fit_args = {}

                mll_cv = botorch.fit_gpytorch_mll(mll_cv, **fit_args)

                if visualize:
                    visual.on_update(model_cv)
                    filename = f"cross_validate{n_cv}.html"
                    with open(filename, "w") as fout:
                        print(f"Writing {filename}")
                        fout.write(visual.full_html())

                with torch.no_grad():
                    posterior = model_cv.posterior(
                        cv_folds.test_X, observation_noise=True
                    )

                cv_results = CVResults(
                    model=model_cv,
                    posterior=posterior,
                    observed_Y=cv_folds.test_Y,
                    observed_Yvar=cv_folds.test_Yvar,
                )

                filename = os.path.join(result_dir, f"cv-{sweep_name}-{model_name}{suffix}-{n_cv}.pt")
                result = {
                    "state_dict": cv_results.model.state_dict(),
                    "posterior": cv_results.posterior,
                    "observed_Y": cv_results.observed_Y.cpu(),
                }
                print(f"Saving {filename}")
                torch.save(result, filename)

            if trace:
                from torch.profiler import profile, record_function, ProfilerActivity
                group_by_shape = False
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                             record_shapes=group_by_shape) as prof:
                    with record_function("batch_cross_validation"):
                        run_cross_validation()
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                print(prof.key_averages(group_by_input_shape=group_by_shape).table(
                    sort_by="cuda_time_total", row_limit=50))
                filename = f"cross_validation_{sweep_name}_{model_name}_{n_cv}.json"
                prof.export_chrome_trace(filename)
                print("Saved", filename)
            else:
                run_cross_validation()


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
