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

import linear_operator

gpytorch.settings.debug._set_state(False)
linear_operator.settings.debug._set_state(False)
# gpytorch.settings.trace_mode._set_state(True)
botorch.settings.debug._set_state(False)


try:
    import nvtx
except ImportError:
    class NVTXAnnotateStub:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args, **kwargs):
            pass

    # replace with stub
    class NVTXStub:
        annotate = NVTXAnnotateStub

    nvtx = NVTXStub


def initialize(sweep_name, model_name, vectorize, torch_compile, num_models=1):
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

    if num_models > 1:
        X = X.expand(num_models, *X.shape)
        Y = Y.expand(num_models, *Y.shape)

    model = model_cls(X, Y).to(device)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    s = f"{sweep_name} {model_name} vectorize={vectorize} torch.compile={torch_compile}"
    if num_models > 1:
        s += f" num_models={num_models}"
    print(s)

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
        with nvtx.annotate("benchmark", color="blue"):
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

    # warmup
    output = model(train_X)
    loss = -mll(output, train_Y.squeeze(-1))
    loss.sum().backward()
    del output
    del loss
    if torch.cuda.is_available():
        torch.cuda.synchronize()

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
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        tend = time.time()
        print(f"Elapsed time: {tend - tstart:>2f}")


def benchmark_covariance_optimize(sweep_name, model_name, train_X, train_Y,
                                  model, mll, search_space, search_xform,
                                  trace=False, repetitions=200):
    """
    Benchmark the covariance kernel forward and backward pass, using input
    shapes that are representative of what would be seen during Bayesian
    Optimization.
    """
    model.eval()

    x1 = torch.randn((60, 2, 26), device=train_X.device,
                     requires_grad=True)
    x2 = torch.randn((60, 381, 26), device=train_X.device,
                     requires_grad=True)

    def f():
        with gpytorch.settings.lazily_evaluate_kernels(False):
            cov = model.covar_module(x1, x2).to_dense()
            loss = cov.sum()
            loss.backward()

    # warmup
    f()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print(f"benchmark_optimize_covariance: covar {x1.shape} {x2.shape}")
    if trace:
        group_by_shape = True
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=group_by_shape) as prof:
            with record_function("optimization_test"):
                for _ in range(repetitions):
                    f()

        print(prof.key_averages(group_by_input_shape=group_by_shape).table(
            sort_by="cuda_time_total", row_limit=50))
        filename = f"benchmark_covariance_optimize_{sweep_name}_{model_name}.json"
        prof.export_chrome_trace(filename)
        print("Saved", filename)

    else:
        # torch.cuda.set_sync_debug_mode(2)
        tstart = time.time()
        for _ in range(repetitions):
            f()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        tend = time.time()
        # torch.cuda.set_sync_debug_mode(0)
        print(f"Elapsed time: {tend - tstart:>2f}")



def benchmark_covariance_fit(sweep_name, model_name, train_X, train_Y,
                             model, mll, search_space, search_xform,
                             trace=False, repetitions=200):
    """
    """
    with nvtx.annotate("warmup", color="red"):
        model.train()

        transformed_train_X = model.transform_inputs(train_X)

        def f():
            with gpytorch.settings.lazily_evaluate_kernels(False):
                cov = model.covar_module(transformed_train_X).to_dense()
                loss = cov.sum()
                loss.backward()

        # warmup
        with gpytorch.settings.lazily_evaluate_kernels(False):
            f()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    X_shape = transformed_train_X.shape
    print(f"benchmark_covariance_fit: covar {X_shape} {X_shape}")
    if trace:
        group_by_shape = False
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=group_by_shape) as prof:
            with record_function("benchmark_covariance_fit"):
                for _ in range(repetitions):
                    f()

        print(prof.key_averages(group_by_input_shape=group_by_shape).table(
            sort_by="cuda_time_total", row_limit=50))
        filename = f"benchmark_covariance_fit_{sweep_name}_{model_name}.json"
        prof.export_chrome_trace(filename)
        print("Saved", filename)
    else:
        with nvtx.annotate("benchmark", color="blue"):
            tstart = time.time()
            for _ in range(repetitions):
                f()

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            tend = time.time()
        print(f"Elapsed time: {tend - tstart:>2f}")


def cv_initialize(sweep_name, model_name, vectorize, torch_compile):
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

    from botorch.cross_validation import gen_loo_cv_folds
    n_cv = 120
    cv_folds = gen_loo_cv_folds(train_X=X[:n_cv], train_Y=Y[:n_cv])

    model = model_cls(cv_folds.train_X, cv_folds.train_Y).to(device)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    print(f"{sweep_name} {model_name} vectorize={vectorize} torch.compile={torch_compile}")

    return cv_folds.train_X, cv_folds.train_Y, model, mll, search_space, search_xform


def benchmark_covariance_cross_validate(
        sweep_name, model_name, train_X,
        train_Y, model, mll, search_space, search_xform, trace=False,
        repetitions=200):
    """
    """
    with nvtx.annotate("warmup", color="red"):
        model.train()

        transformed_train_X = model.transform_inputs(train_X)

        def f():
            with gpytorch.settings.lazily_evaluate_kernels(False):
                cov = model.covar_module(transformed_train_X).to_dense()
                loss = cov.sum()
                loss.backward()

        # warmup
        with gpytorch.settings.lazily_evaluate_kernels(False):
            f()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    X_shape = transformed_train_X.shape
    print(f"benchmark_covariance_cross_validate: covar {X_shape} {X_shape}")
    if trace:
        group_by_shape = True
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     # record_shapes=group_by_shape
                     ) as prof:
            with record_function("benchmark_covariance_cross_validate"):
                for _ in range(repetitions):
                    f()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        print(prof.key_averages(group_by_input_shape=group_by_shape).table(
            sort_by="cuda_time_total", row_limit=50))
        filename = f"benchmark_covariance_cross_validate_{sweep_name}_{model_name}.json"
        prof.export_chrome_trace(filename)
        print("Saved", filename)

    else:
        # torch.cuda.set_sync_debug_mode(2)
        with nvtx.annotate("benchmark", color="blue"):
            tstart = time.time()
            for _ in range(repetitions):
                f()

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            tend = time.time()
        # torch.cuda.set_sync_debug_mode(0)
        print(f"Elapsed time: {tend - tstart:>2f}")


def benchmark_gp_cross_validate(
        sweep_name, model_name, train_X,
        train_Y, model, mll, search_space, search_xform, trace=False,
        repetitions=200):
    """
    """
    with nvtx.annotate("warmup", color="red"):
        model.train()

        # grad_log = []

        def f():
            with gpytorch.settings.lazily_evaluate_kernels(False):
                output = model(train_X)
                loss = -mll(output, train_Y.squeeze(-1))
                loss.sum().backward()
                # grad_log.append([t.grad.cpu().clone() for t in list(model.parameters())])

        # warmup
        with gpytorch.settings.lazily_evaluate_kernels(False):
            f()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    X_shape = train_X.shape
    print(f"benchmark_gp_cross_validate: covar {X_shape} {X_shape}")
    if trace:
        group_by_shape = True
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=group_by_shape) as prof:
            with record_function("benchmark_gp_cross_validate"):
                for _ in range(repetitions):
                    f()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        print(prof.key_averages(group_by_input_shape=group_by_shape).table(
            sort_by="cuda_time_total", row_limit=50))
        filename = f"benchmark_gp_cross_validate_{sweep_name}_{model_name}.json"
        prof.export_chrome_trace(filename)
        print("Saved", filename)

    else:
        # torch.cuda.set_sync_debug_mode(2)
        with nvtx.annotate("benchmark", color="blue"):
            tstart = time.time()
            for _ in range(repetitions):
                f()

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            tend = time.time()
        # torch.cuda.set_sync_debug_mode(0)
        print(f"Elapsed time: {tend - tstart:>2f}")

    # device_str = ("cuda" if torch.cuda.is_available() else "cpu")
    # filename = f"grad_log_{sweep_name}_{model_name}_{device_str}.pt"
    # print("Saving", filename)
    # torch.save(grad_log, filename)


def benchmark_optimize(sweep_name, model_name, train_X, train_Y, model, mll,
                       search_space, search_xform, trace=False, repetitions=200):
    with nvtx.annotate("warmup", color="red"):
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
        if torch.cuda.is_available():
            torch.cuda.synchronize()

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
        with nvtx.annotate("benchmark", color="blue"):
            tstart = time.time()
            for _ in range(repetitions):
                posterior = model.posterior(X)
                loss = posterior.mean.sum()
                loss.backward()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            tend = time.time()
        # torch.cuda.set_sync_debug_mode(0)
        print(f"Elapsed time: {tend - tstart:>2f}")


def scenario_optimize(sweep_name, model_name, train_X, train_Y, model, mll,
                      search_space, search_xform, trace=False,
                      force_retrain=False):
    filename = f"performance_test_{sweep_name}_{model_name}.pt"
    retrain = force_retrain or not os.path.exists(filename)
    if retrain:
        with nvtx.annotate("fitting", color="red"):
            print("fitting")
            botorch.fit_gpytorch_model(mll)
            print("fitted")
            torch.save(mll.state_dict(), filename)
    else:
        mll.train()
        mll.load_state_dict(torch.load(filename))
        with nvtx.annotate("warmup", color="red"):
            posterior = model.posterior(train_X.unsqueeze(1))
            loss = posterior.mean.sum()
            loss.backward()
            del posterior
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    rounding_function = ol.transforms.UntransformThenTransform(
        search_space, search_xform,
    ).transform

    print(f"scenario_optimize: {sweep_name} {model_name}")
    model.eval()

    def f():
        candidates, acq_value = botorch.optim.optimize_acqf_mixed(
            acq_function=botorch.acquisition.qNoisyExpectedImprovement(
                model=model,
                X_baseline=train_X,
                sampler=botorch.sampling.SobolQMCNormalSampler(
                    sample_shape=torch.Size([256])
                    # sample_shape=torch.Size([64])
                ),
                objective=botorch.acquisition.GenericMCObjective(
                            lambda Z: -Z[..., 0]),
            ),
            bounds=ol.botorch_bounds(search_space).to(train_X.device),
            fixed_features_list=ol.all_base_configurations(search_space),
            q=1,
            num_restarts=60,
            raw_samples=1024,
            # num_restarts=10, #60,
            # raw_samples=64, #1024,
            # inequality_constraints=ol.botorch_inequality_constraints(config["constraints"]),
            post_processing_func=rounding_function,
        )

        return candidates, acq_value

    if trace:
        group_by_shape = True
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=group_by_shape) as prof:
            with record_function("optimization_test"):
                f()
        print(prof.key_averages(group_by_input_shape=group_by_shape).table(
            sort_by="cuda_time_total", row_limit=50))
        filename = f"scenario_optimize_{sweep_name}_{model_name}.json"
        prof.export_chrome_trace(filename)
        print("Saved", filename)
    else:
        with nvtx.annotate("benchmark", color="blue"):
            tstart = time.time()
            candidates, acq_value = f()
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
    parser.add_argument("--num-models", type=int, default=1)

    parser.add_argument("--test", type=str, default="benchmark_fit", choices=[
        "benchmark_fit", "benchmark_optimize", "benchmark_covariance_optimize",
        "benchmark_covariance_fit", "scenario_fit", "scenario_optimize",
        "benchmark_covariance_cross_validate", "benchmark_gp_cross_validate"])

    args = parser.parse_args()

    print("gpytorch debug:", gpytorch.settings.debug.on())
    print("linear_operator debug:", linear_operator.settings.debug.on())
    print("botorch debug:", botorch.settings.debug.on())

    if args.test == "benchmark_fit":
        benchmark_fit(args.sweep_name, args.model_name,
                      *initialize(args.sweep_name, args.model_name,
                                  vectorize=args.vectorize,
                                  torch_compile=args.compile,
                                  num_models=args.num_models),
                      trace=args.trace, repetitions=args.repetitions)
    elif args.test == "benchmark_optimize":
        benchmark_optimize(args.sweep_name, args.model_name,
                           *initialize(args.sweep_name, args.model_name,
                                       vectorize=args.vectorize,
                                       torch_compile=args.compile,
                                       num_models=args.num_models),
                           trace=args.trace, repetitions=args.repetitions)

    elif args.test == "benchmark_covariance_optimize":
        benchmark_covariance_optimize(args.sweep_name, args.model_name,
                                      *initialize(args.sweep_name, args.model_name,
                                                  vectorize=args.vectorize,
                                                  torch_compile=args.compile,
                                                  num_models=args.num_models),
                                      trace=args.trace, repetitions=args.repetitions)
    elif args.test == "benchmark_covariance_fit":
        benchmark_covariance_fit(args.sweep_name, args.model_name,
                                 *initialize(args.sweep_name, args.model_name,
                                             vectorize=args.vectorize,
                                             torch_compile=args.compile,
                                             num_models=args.num_models),
                                 trace=args.trace, repetitions=args.repetitions)
    elif args.test == "benchmark_covariance_cross_validate":
        benchmark_covariance_cross_validate(
            args.sweep_name, args.model_name,
            *cv_initialize(args.sweep_name, args.model_name,
                           vectorize=args.vectorize,
                           torch_compile=args.compile),
            trace=args.trace, repetitions=args.repetitions)
    elif args.test == "benchmark_gp_cross_validate":
        benchmark_gp_cross_validate(
            args.sweep_name, args.model_name,
            *cv_initialize(args.sweep_name, args.model_name,
                           vectorize=args.vectorize,
                           torch_compile=args.compile),
            trace=args.trace, repetitions=args.repetitions)
    elif args.test == "scenario_fit":
        scenario_fit(args.sweep_name, args.model_name,
                     *initialize(args.sweep_name, args.model_name,
                                 vectorize=args.vectorize,
                                 torch_compile=args.compile,
                                 num_models=args.num_models),
                     trace=args.trace)
    elif args.test == "scenario_optimize":
        scenario_optimize(args.sweep_name, args.model_name,
                          *initialize(args.sweep_name, args.model_name,
                                      vectorize=args.vectorize,
                                      torch_compile=args.compile,
                                      num_models=args.num_models),
                          trace=args.trace, force_retrain=args.force_retrain)


if __name__ == "__main__":
    main()
