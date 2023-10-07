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

# import torch._dynamo
# torch._dynamo.config.verbose = True

gpytorch.settings.debug._set_state(False)
# gpytorch.settings.trace_mode._set_state(True)
botorch.settings.debug._set_state(False)


def initialize(sweep_name, model_name, vectorize, torch_compile):
    torch.set_default_dtype(torch.float64)

    config = CONFIGS[sweep_name]
    model_cls = MODELS[model_name]

    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)
    sweep_dir = os.path.join(project_dir, "results", sweep_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    search_space = config["search_space"]
    search_xform = config["search_xform"].to(device)

    model_cls = partial(model_cls,
                        search_space=search_space,
                        search_xform=search_xform,
                        round_inputs=False,
                        vectorize=vectorize,
                        torch_compile=torch_compile)

    configs, trial_dirs, _ = parse_results(sweep_name)

    X, Y = configs_dirs_to_X_Y(configs, trial_dirs, trial_dir_to_loss_y,
                               config["parameter_space"],
                               search_xform,
                               device=device)

    model = model_cls(X, Y).to(device)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    print(f"{sweep_name} {model_name} vectorize={vectorize} torch.compile={torch_compile}")

    return X, Y, model, mll, search_space, search_xform


def scenario_fit(sweep_name, model_name, train_X, train_Y, model, mll,
                 search_space, search_xform, trace=False):
    mll.train()

    group_by_shape = False

    print(f"scenario_fit: {sweep_name} {model_name}")
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


def benchmark_fit(sweep_name, model_name, train_X, train_Y, model, mll,
                  search_space, search_xform, trace=False, repetitions=200):
    mll.train()

    group_by_shape = False

    print(f"benchmark_fit: {sweep_name} {model_name}")
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


def benchmark_optimize(sweep_name, model_name, train_X, train_Y, model, mll,
                       search_space, search_xform, trace=False, repetitions=200):
    X = train_X.clone()
    Y = train_Y.clone()

    # This tests n different sets of candidates in parallel
    X = X.unsqueeze(1)
    # X = X.repeat(12, 1).unsqueeze(1)

    X.requires_grad_(True)
    model.eval()

    # warmup
    posterior = model.posterior(X)
    loss = posterior.mean.sum()
    loss.backward()
    del posterior
    del loss

    print(f"benchmark_optimize: candidates size {X.shape}")
    if trace:
        group_by_shape = True
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=group_by_shape) as prof:
            with record_function("optimization_test"):
                for _ in range(repetitions):
                    posterior = model.posterior(X)
                    loss = posterior.mean.sum()
                    loss.backward()

        print(prof.key_averages(group_by_input_shape=group_by_shape).table(
            sort_by="cuda_time_total", row_limit=50))
        filename = f"optimization_test_{sweep_name}_{model_name}.json"
        prof.export_chrome_trace(filename)
        print("Saved", filename)

    else:
        # torch.cuda.set_sync_debug_mode(2)
        tstart = time.time()
        for _ in range(repetitions):
            posterior = model.posterior(X)
            loss = posterior.mean.sum()
            loss.backward()

        tend = time.time()
        # torch.cuda.set_sync_debug_mode(0)
        print(f"Elapsed time: {tend - tstart:>2f}")


def scenario_optimize(sweep_name, model_name, train_X, train_Y, model, mll,
                      search_space, search_xform, trace=False,
                      force_retrain=False):
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
        search_space, search_xform,
    ).transform

    print(f"scenario_optimize: {sweep_name} {model_name}")
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
        bounds=ol.botorch_bounds(search_space).to(train_X.device),
        fixed_features_list=ol.all_base_configurations(search_space),
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
    parser.add_argument("--vectorize", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--force-retrain", action="store_true")

    parser.add_argument("--test", type=str, default="benchmark_fit", choices=[
        "benchmark_fit", "benchmark_optimize", "scenario_fit", "scenario_optimize"]
    )

    args = parser.parse_args()

    print("gpytorch debug:", gpytorch.settings.debug.on())
    print("botorch debug:", botorch.settings.debug.on())

    if args.test == "benchmark_fit":
        benchmark_fit(args.sweep_name, args.model_name,
                      *initialize(args.sweep_name, args.model_name,
                                  vectorize=args.vectorize,
                                  torch_compile=args.compile),
                      trace=args.trace, repetitions=args.repetitions)
    elif args.test == "benchmark_optimize":
        benchmark_optimize(args.sweep_name, args.model_name,
                           *initialize(args.sweep_name, args.model_name,
                                       vectorize=args.vectorize,
                                       torch_compile=args.compile),
                           trace=args.trace, repetitions=args.repetitions)
    elif args.test == "scenario_fit":
        scenario_fit(args.sweep_name, args.model_name,
                     *initialize(args.sweep_name, args.model_name,
                                 vectorize=args.vectorize,
                                 torch_compile=args.compile),
                     trace=args.trace)
    elif args.test == "scenario_optimize":
        scenario_optimize(args.sweep_name, args.model_name,
                          *initialize(args.sweep_name, args.model_name,
                                      vectorize=args.vectorize,
                                      torch_compile=args.compile),
                          trace=args.trace, force_retrain=args.force_retrain)


if __name__ == "__main__":
    main()
