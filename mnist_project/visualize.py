import base64
import csv
import io

import botorch
import gpytorch
import numpy as np
import torch
import vexpr as vp
import vexpr.web

from src.gp import MODELS
from src.gp.gp_utils import (
    configs_dirs_to_X_Y,
)
from src.gp.mnist_metrics import trial_dir_to_loss_y
from src.scheduling import parse_results
from src.sweeps import CONFIGS


def aliased_kernel(model):
    with torch.no_grad():
        parameters = {name: module.value
                      for name, module in model.covar_module.state.items()}

    expr = vp.partial_eval(model.kernel_viz_vexpr, parameters)
    aliased_expr, aliases, values = vexpr.web.alias_values(expr)
    return aliased_expr, aliases, values


def headers(kernel_keys):
    return ["mean", "noise"] + kernel_keys


def row(model, precomputed_kernel_values=None):
    if precomputed_kernel_values is None:
        _, _, kernel_values = aliased_kernel(model)
    else:
        kernel_values = precomputed_kernel_values

    result = [
        str(model.mean_module.constant.detach().item()),
        str(model.likelihood.noise.detach().item()),
    ] + kernel_values

    return base64.b64encode(
        np.array(result, dtype=np.float32).tobytes()
    ).decode('utf-8')


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

    kernel_structure, kernel_keys, initial_values = aliased_kernel(model)
    outfile = io.StringIO()
    print(row(model, precomputed_kernel_values=initial_values),
          file=outfile)

    del initial_values

    def callback(parameters, result):
        """
        Note: botorch will wrap this callback in slow code
        """
        print(row(model), file=outfile)

    botorch.fit_gpytorch_mll(mll, optimizer_kwargs=dict(callback=callback))
    print(row(model), file=outfile)

    style = "border: 1px solid silver; border-radius: 5px; padding: 10px; overflow: auto;"
    html_contents = vexpr.web.full_html(
        vexpr.web.visualize_timeline_html(
            html_preamble=lambda element_id: f"""
            <style>
            .vexpr-code {{
                color: gray;
            }}
            </style>
            <div style="width:800px; padding: 10px;" id="{element_id}">
            <div class="timesteps"></div>
            <p><strong>Mean:</strong> constant</p>
            <pre class="mean" style="{style} height: 50px;"></pre>
            <p><strong>Covariance:</strong> Start with matrix formed by kernel</p>
            <pre class="kernel" style="{style} height: 400px;"></pre>
            <p>Then take that matrix and add the following number to each value along the diagonal</p>
            <pre class="noise" style="{style} height: 20px;"></pre>
            </div>
            """,
            components=[vexpr.web.time_control(class_name="timesteps"),
                        vexpr.web.position_view(class_name="mean", key="mean"),
                        vexpr.web.expression_view(class_name="kernel",
                                                   keys=kernel_keys,
                                                   text=repr(kernel_structure)),
                        vexpr.web.scalar_view(class_name="noise", key="noise")],
            headers=headers(kernel_keys),
            encoded_data=outfile.getvalue()
        )
    )


    filename = "newly_created.html"
    with open(filename, "w") as fout:
        print(f"Writing {filename}")
        fout.write(html_contents)




scenario_fit("mnist1", "VexprHandsOnVisualizedGP",
             vectorize=True, torch_compile=False,
             num_models=1)
