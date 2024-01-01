import botorch
import gpytorch
import torch
import vexpr as vp

from src.gp import MODELS
from src.gp.gp_utils import (
    configs_dirs_to_X_Y,
)
from src.gp.mnist_metrics import trial_dir_to_loss_y
from src.scheduling import parse_results
from src.sweeps import CONFIGS
from src.visuals import MeanNoiseKernelTimeline


def scenario_fit(sweep_name, model_name, vectorize, torch_compile, num_models=1):
    torch.set_default_dtype(torch.float64)

    config = CONFIGS[sweep_name]
    model_cls = MODELS[model_name]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    search_space = config["search_space"]
    search_xform = config["search_xform"].to(device)

    configs, trial_dirs, _ = parse_results(sweep_name)

    X, Y = configs_dirs_to_X_Y(configs, trial_dirs, trial_dir_to_loss_y,
                               config["parameter_space"],
                               search_xform,
                               device=device)

    if num_models > 1:
        X = X.expand(num_models, *X.shape)
        Y = Y.expand(num_models, *Y.shape)

    model = model_cls(X, Y, search_space=search_space,
                      search_xform=search_xform,
                      round_inputs=False,
                      vectorize=vectorize,
                      torch_compile=torch_compile,
                      visualize=False).to(device)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    s = f"{sweep_name} {model_name} vectorize={vectorize} torch.compile={torch_compile}"
    if num_models > 1:
        s += f" num_models={num_models}"

    mll.train()

    visual = MeanNoiseKernelTimeline(model)
    iteration = [0]

    def callback(parameters, result):
        """
        Note: botorch will wrap this callback in slow code
        """
        iteration[0] += 1
        visual.on_update(model, iteration[0])

    botorch.fit_gpytorch_mll(mll, optimizer_kwargs=dict(callback=callback))

    filename = "newly_created.html"
    with open(filename, "w") as fout:
        print(f"Writing {filename}")
        fout.write(visual.full_html())


scenario_fit("mnist1", "VexprHandsOnVisualizedGP",
             vectorize=True, torch_compile=False,
             num_models=1)
