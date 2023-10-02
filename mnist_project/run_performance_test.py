import os
import time
from functools import partial

import torch
import botorch
import gpytorch
from torch.profiler import profile, record_function, ProfilerActivity

import outerloop as ol

from src.gp import MODELS
from src.gp.gp_utils import configs_dirs_to_X_Y
from src.gp.mnist_metrics import trial_dir_to_loss_y
from src.scheduling import parse_results
from src.sweeps import CONFIGS

gpytorch.settings.debug._set_state(False)
botorch.settings.debug._set_state(False)


def initialize(sweep_name, model_name):
    torch.set_default_dtype(torch.float64)

    config = CONFIGS[sweep_name]
    model_cls = MODELS[model_name]

    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)
    sweep_dir = os.path.join(project_dir, "results", sweep_name)

    model_cls = partial(model_cls,
                        search_space=config["search_space"],
                        search_xform=config["search_xform"],
                        round_inputs=False)

    configs, trial_dirs, _ = parse_results(sweep_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, Y = configs_dirs_to_X_Y(configs, trial_dirs, trial_dir_to_loss_y,
                               config["parameter_space"],
                               config["search_xform"],
                               device=device)

    model = model_cls(X, Y).to(device)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    return X, Y, model, mll, config


def scenario_fit(sweep_name, model_name, trace=False):
    train_X, train_Y, model, mll, config = initialize(sweep_name, model_name)
    mll.train()

    group_by_shape = False

    if trace:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=group_by_shape) as prof:
            with record_function("optimization_test"):
                botorch.fit_gpytorch_mll(mll)

        print(f"model {model_name}")
        print(prof.key_averages(group_by_input_shape=group_by_shape).table(
            sort_by="cpu_time_total", row_limit=20))
        prof.export_chrome_trace(f"perf-fit-{sweep_name}-{model_name}.json")
    else:
        # last_result = [None]
        # def callback(parameters, result):
        #     """
        #     Note: botorch will wrap this callback in slow code, so it is
        #     disabled by default.
        #     """
        #     last_result[0] = result

        tstart = time.time()
        botorch.fit_gpytorch_mll(mll,
                                 # optimizer_kwargs=dict(
                                 #     callback=callback
                                 # )
                                 )
        tend = time.time()
        elapsed = tend - tstart
        print(f"Elapsed time: {elapsed:>2f}")
        # print(last_result[0])
        # print(f"Mean time per iteration: {elapsed / last_result[0].step:>2f}")


def benchmark_fit(sweep_name, model_name, trace=False, repetitions=200):
    train_X, train_Y, model, mll, config = initialize(sweep_name, model_name)
    mll.train()

    group_by_shape = False

    print("training")
    if trace:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=group_by_shape) as prof:
            with record_function("train_test"):
                for _ in range(repetitions):
                    output = model(train_X)
                    loss = -mll(output, train_Y.squeeze(-1))
                    loss.sum().backward()

        print(prof.key_averages(group_by_input_shape=group_by_shape).table(
            sort_by="cpu_time_total", row_limit=20))
        prof.export_chrome_trace(f"perf-train-{sweep_name}-{model_name}.json")
    else:
        tstart = time.time()
        for _ in range(repetitions):
            # torch.cuda.set_sync_debug_mode(2)
            output = model(train_X)
            # torch.cuda.set_sync_debug_mode(0)
            loss = -mll(output, train_Y.squeeze(-1))
            loss.sum().backward()
        tend = time.time()
        print(f"Elapsed time: {tend - tstart:>2f}")


def benchmark_optimize(sweep_name, model_name, trace=False, repetitions=200):
    train_X, train_Y, model, mll, config = initialize(sweep_name, model_name)

    X = train_X.clone()
    Y = train_Y.clone()

    # X = X.repeat(12, 1).unsqueeze(1)
    X = X.unsqueeze(1)
    X.requires_grad_(True)
    model.eval()

    group_by_shape = False

    print("optimization")
    print(f"model {model_name}, candidates size {X.shape}")
    if trace:

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=group_by_shape) as prof:
            with record_function("optimization_test"):
                for _ in range(repetitions):
                    posterior = model.posterior(X)
                    loss = posterior.mean.sum() - posterior.variance.sum()
                    loss.backward()

        print(prof.key_averages(group_by_input_shape=group_by_shape).table(
            sort_by="cpu_time_total", row_limit=20))
        prof.export_chrome_trace(f"optimization_test_{sweep_name}_{model_name}.json")

    else:
        tstart = time.time()
        for _ in range(repetitions):
            posterior = model.posterior(X)
            loss = posterior.mean.sum() - posterior.variance.sum()
            # torch.cuda.set_sync_debug_mode(2)
            loss.backward()
            # torch.cuda.set_sync_debug_mode(0)
        tend = time.time()
        print(f"Elapsed time: {tend - tstart:>2f}")


def scenario_optimize(sweep_name, model_name, trace=False, force_retrain=False):
    train_X, train_Y, model, mll, config = initialize(sweep_name, model_name)

    filename = f"performance_test_{sweep_name}_{model_name}.pt"
    retrain = force_retrain or not os.path.exists(filename)
    if retrain:
        print("fitting")
        botorch.fit_gpytorch_model(mll)
        print("fitted")
        torch.save(mll.state_dict(), filename)
    else:
        mll.train()
        mll.load_state_dict(torch.load(filename))

    rounding_function = ol.transforms.UntransformThenTransform(
        config["search_space"], config["search_xform"]
    ).transform

    model.eval()
    tstart = time.time()
    candidates, acq_value = botorch.optim.optimize_acqf_mixed(
        acq_function=botorch.acquisition.qNoisyExpectedImprovement(
                    model=model,
                    X_baseline=train_X,
                    sampler=botorch.sampling.SobolQMCNormalSampler(
                        sample_shape=torch.Size([64])
                    ),
                    objective=botorch.acquisition.GenericMCObjective(
                        lambda Z: -Z[..., 0])
                ),
        bounds=ol.botorch_bounds(config["search_space"]),
        fixed_features_list=ol.all_base_configurations(config["search_space"]),
        q=1,
        num_restarts=10, #60,
        raw_samples=64, #1024,
        # inequality_constraints=ol.botorch_inequality_constraints(config["constraints"]),
        post_processing_func=rounding_function,
    )
    tend = time.time()
    print(f"Elapsed time: {tend - tstart:>2f}, acq_value: {acq_value}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="VexprHandsOnLossModel")
    parser.add_argument("--sweep-name", type=str, default="mnist1")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--force-retrain", action="store_true")

    parser.add_argument("--test", type=str, default="benchmark_fit", choices=[
        "benchmark_fit", "benchmark_optimize", "scenario_fit", "scenario_optimize"]
    )

    cmd_args = parser.parse_args()

    print("gpytorch debug:", gpytorch.settings.debug.on())
    print("botorch debug:", botorch.settings.debug.on())

    if cmd_args.test == "benchmark_fit":
        benchmark_fit(sweep_name=cmd_args.sweep_name, model_name=cmd_args.model_name,
                      trace=cmd_args.trace, repetitions=cmd_args.repetitions)
    elif cmd_args.test == "benchmark_optimize":
        benchmark_optimize(sweep_name=cmd_args.sweep_name, model_name=cmd_args.model_name,
                           trace=cmd_args.trace, repetitions=cmd_args.repetitions)
    elif cmd_args.test == "scenario_fit":
        scenario_fit(sweep_name=cmd_args.sweep_name, model_name=cmd_args.model_name,
                     trace=cmd_args.trace)
    elif cmd_args.test == "scenario_optimize":
        scenario_optimize(sweep_name=cmd_args.sweep_name, model_name=cmd_args.model_name,
                          trace=cmd_args.trace, force_retrain=cmd_args.force_retrain)


if __name__ == "__main__":
    main()
