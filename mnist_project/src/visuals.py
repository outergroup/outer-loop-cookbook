import base64
import io

import numpy as np
import torch
import vexpr as vp
import vexpr.notebook
import vexpr.web


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
        model.mean_module.constant.detach().tolist(),
        model.likelihood.noise.squeeze(-1).detach().tolist(),
    ] + kernel_values

    return base64.b64encode(
        np.array(result, dtype=np.float32).tobytes()
    ).decode('utf-8')


def common_visualize_kwargs(outfile):
    style = "border: 1px solid silver; border-radius: 5px; padding: 10px; overflow: auto;"
    return dict(
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
        encoded_data=outfile.getvalue(),
    )


def scalar_visualize_kwargs(kernel_keys, kernel_structure, outfile):
    return dict(
        components=[vexpr.web.time_control(class_name="timesteps"),
                    vexpr.web.position_view(class_name="mean", key="mean"),
                    vexpr.web.expression_view(class_name="kernel",
                                              keys=kernel_keys,
                                              text=repr(kernel_structure)),
                    vexpr.web.scalar_view(class_name="noise", key="noise")],
        encoded_to_timesteps=vexpr.web.scalar_timesteps_js(headers(kernel_keys)),
        **common_visualize_kwargs(outfile)
    )


class MeanNoiseKernelVisual:
    def __init__(self, model):
        self.outfile = io.StringIO()
        (self.kernel_structure,
         self.kernel_keys,
         initial_values) = aliased_kernel(model)
        print(row(model, precomputed_kernel_values=initial_values),
              file=self.outfile)

    def on_update(self, model):
        print(row(model), file=self.outfile)

    def full_html(self):
        return vexpr.web.full_html(
            vexpr.web.visualize_timeline_html(
                **scalar_visualize_kwargs(self.kernel_keys, self.kernel_structure,
                                          self.outfile)
            )
        )


class MeanNoiseKernelNotebookVisual:
    def __init__(self, model):
        self.outfile = io.StringIO()
        (self.kernel_structure,
         self.kernel_keys,
         initial_values) = aliased_kernel(model)
        print(row(model, precomputed_kernel_values=initial_values),
              file=self.outfile)

        self.element_id = vexpr.notebook.visualize_timeline(
            **scalar_visualize_kwargs(self.kernel_keys, self.kernel_structure,
                                      self.outfile))

    def on_update(self, model):
        print(row(model), file=self.outfile)
        vexpr.notebook.update_timeline(self.element_id, self.outfile.getvalue())


def scalar_distribution_visualize_kwargs(kernel_keys, kernel_structure, outfile,
                                         num_values_per_param):
    return dict(
        components=[
            vexpr.web.time_control(class_name="timesteps"),
            vexpr.web.position_distribution_view(class_name="mean",
                                                 key="mean"),
            vexpr.web.expression_distribution_view(
                class_name="kernel",
                keys=kernel_keys,
                text=repr(kernel_structure)),
            vexpr.web.scalar_distribution_view(class_name="noise", key="noise")],
        encoded_to_timesteps=vexpr.web.scalar_distribution_timesteps_js(
            headers(kernel_keys), num_values_per_param),
        **common_visualize_kwargs(outfile)
    )


class MeanNoiseKernelDistributionVisual:
    def __init__(self, model, num_values_per_param):
        self.outfile = io.StringIO()
        (self.kernel_structure,
         self.kernel_keys,
         initial_values) = aliased_kernel(model)
        self.num_values_per_param = num_values_per_param
        print(row(model, precomputed_kernel_values=initial_values),
              file=self.outfile)

    def on_update(self, model):
        print(row(model), file=self.outfile)

    def full_html(self):
        return vexpr.web.full_html(
            vexpr.web.visualize_timeline_html(
                **scalar_distribution_visualize_kwargs(self.kernel_keys,
                                                       self.kernel_structure,
                                                       self.outfile,
                                                       self.num_values_per_param),
            )
        )


class MeanNoiseKernelDistributionNotebookVisual:
    def __init__(self, model, num_values_per_param):
        self.outfile = io.StringIO()
        (self.kernel_structure,
         self.kernel_keys,
         initial_values) = aliased_kernel(model)
        self.num_values_per_param = num_values_per_param
        print(row(model, precomputed_kernel_values=initial_values),
              file=self.outfile)

        self.element_id = vexpr.notebook.visualize_timeline(
            **scalar_distribution_visualize_kwargs(self.kernel_keys,
                                                   self.kernel_structure,
                                                   self.outfile,
                                                   self.num_values_per_param))

    def on_update(self, model):
        print(row(model), file=self.outfile)
        vexpr.notebook.update_timeline(self.element_id, self.outfile.getvalue())
