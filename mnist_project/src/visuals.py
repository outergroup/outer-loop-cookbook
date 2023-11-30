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
        str(model.mean_module.constant.detach().item()),
        str(model.likelihood.noise.detach().item()),
    ] + kernel_values

    return base64.b64encode(
        np.array(result, dtype=np.float32).tobytes()
    ).decode('utf-8')



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
        style = "border: 1px solid silver; border-radius: 5px; padding: 10px; overflow: auto;"
        return vexpr.web.full_html(
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
                                                      keys=self.kernel_keys,
                                                      text=repr(self.kernel_structure)),
                            vexpr.web.scalar_view(class_name="noise", key="noise")],
                headers=headers(self.kernel_keys),
                encoded_data=self.outfile.getvalue()
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

        style = "border: 1px solid silver; border-radius: 5px; padding: 10px; overflow: auto;"
        self.element_id = vexpr.notebook.visualize_timeline(
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
                                                      keys=self.kernel_keys,
                                                      text=repr(self.kernel_structure)),
                            vexpr.web.scalar_view(class_name="noise", key="noise")],
                headers=headers(self.kernel_keys),
                encoded_data=self.outfile.getvalue()
            )


    def on_update(self, model):
        print(row(model), file=self.outfile)
        vexpr.notebook.update_timeline(self.element_id, self.outfile.getvalue())
