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


def sampling_row(models, samples_per_model):
    all_results = None

    for model in models:
        _, _, kernel_values = aliased_kernel(model)
        kernel_values = [values[:samples_per_model]
                         for values in kernel_values]

        result = [
            model.mean_module.constant.detach()[:samples_per_model].tolist(),
            model.likelihood.noise.squeeze(-1).detach()[:samples_per_model].tolist(),
        ] + kernel_values
        if all_results is None:
            all_results = result
        else:
            all_results = [a + b for a, b in zip(all_results, result)]

    return base64.b64encode(
        np.array(all_results, dtype=np.float32).tobytes()
    ).decode('utf-8')



def snapshot_common_visualize_kwargs(serialized_values):
    return dict(
        html_preamble=lambda element_id: f"""
<style>
.vexpr-code {{
color: gray;
}}
circle.point {{
fill: blue;
opacity: 0.2;
}}
</style>
<div id="{element_id}">
<p><strong>Mean:</strong> constant</p>
<pre class="mean"></pre>
<p><strong>Covariance:</strong> Start with matrix formed by kernel</p>
<pre class="kernel" style="border: 1px solid silver; border-radius: 5px; padding: 10px; overflow: auto; height: 400px;"></pre>
<p><strong>Noise:</strong> Take that matrix and add the following number to each value along the diagonal. <em>(Plotted on log scale.)</em></p>
<pre class="noise"></pre>
</div>""",
        encoded_data=serialized_values,
    )


def timeline_common_visualize_kwargs(outfile):
    return dict(
        html_preamble=lambda element_id: f"""
<style>
.vexpr-code {{
color: gray;
}}
circle.point {{
fill: blue;
opacity: 0.2;
}}
</style>
<div style="width:800px; padding: 10px;" id="{element_id}">
<div class="timesteps"></div>
<p><strong>Mean:</strong> constant</p>
<pre class="mean"></pre>
<p><strong>Covariance:</strong> Start with matrix formed by kernel</p>
<pre class="kernel" style="border: 1px solid silver; border-radius: 5px; padding: 10px; overflow: auto; height: 400px;"></pre>
<p><strong>Noise:</strong> Take that matrix and add the following number to each value along the diagonal. <em>(Plotted on log scale.)</em></p>
<pre class="noise"></pre>
</div>""",
        encoded_data=outfile.getvalue(),
    )


def snapshot_scalar_visualize_kwargs(kernel_keys, kernel_structure, serialized_values):
    return dict(
        components=[vexpr.web.position_view(class_name="mean", key="mean"),
                    vexpr.web.expression_view(class_name="kernel",
                                              keys=kernel_keys,
                                              text=repr(kernel_structure)),
                    vexpr.web.scalar_view(class_name="noise", key="noise")],
        encoded_to_snapshot=vexpr.web.scalar_snapshot_js(headers(kernel_keys)),
        **snapshot_common_visualize_kwargs(serialized_values)
    )


def timeline_scalar_visualize_kwargs(kernel_keys, kernel_structure, outfile):
    return dict(
        components=[vexpr.web.time_control(class_name="timesteps"),
                    vexpr.web.position_view(class_name="mean", key="mean"),
                    vexpr.web.expression_view(class_name="kernel",
                                              keys=kernel_keys,
                                              text=repr(kernel_structure)),
                    vexpr.web.scalar_view(class_name="noise", key="noise")],
        encoded_to_timesteps=vexpr.web.scalar_timesteps_js(headers(kernel_keys)),
        **timeline_common_visualize_kwargs(outfile)
    )


class MeanNoiseKernelTimeline:
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
                **timeline_scalar_visualize_kwargs(self.kernel_keys, self.kernel_structure,
                                                   self.outfile)
            )
        )


class MeanNoiseKernelNotebookSnapshot:
    def __init__(self, model):
        (kernel_structure,
         kernel_keys,
         initial_values) = aliased_kernel(model)
        serialized_values = row(model, precomputed_kernel_values=initial_values)

        self.element_id = vexpr.notebook.visualize_snapshot(
            **snapshot_scalar_visualize_kwargs(kernel_keys, kernel_structure,
                                               serialized_values))

    def on_update(self, model):
        serialized_values = row(model)
        vexpr.notebook.update_snapshot(self.element_id, serialized_values)


class MeanNoiseKernelTimeline:
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
                **snapshot_scalar_visualize_kwargs(self.kernel_keys, self.kernel_structure,
                                                   self.outfile)
            )
        )


class MeanNoiseKernelNotebookTimeline:
    def __init__(self, model):
        self.outfile = io.StringIO()
        (self.kernel_structure,
         self.kernel_keys,
         initial_values) = aliased_kernel(model)
        print(row(model, precomputed_kernel_values=initial_values),
              file=self.outfile)

        self.element_id = vexpr.notebook.visualize_timeline(
            **timeline_scalar_visualize_kwargs(self.kernel_keys, self.kernel_structure,
                                               self.outfile))

    def on_update(self, model):
        print(row(model), file=self.outfile)
        vexpr.notebook.update_timeline(self.element_id, self.outfile.getvalue())


def timeline_distribution_visualize_kwargs(kernel_keys, kernel_structure, outfile,
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
        **timeline_common_visualize_kwargs(outfile)
    )


def snapshot_distribution_visualize_kwargs(kernel_keys, kernel_structure, serialized_values,
                                           num_values_per_param):
    return dict(
        components=[vexpr.web.position_distribution_view(class_name="mean", key="mean"),
                    vexpr.web.expression_distribution_view(class_name="kernel",
                                                           keys=kernel_keys,
                                                           text=repr(kernel_structure)),
                    vexpr.web.scalar_distribution_view(class_name="noise", key="noise")],
        encoded_to_snapshot=vexpr.web.scalar_distribution_snapshot_js(headers(kernel_keys),
                                                                      num_values_per_param),
        **snapshot_common_visualize_kwargs(serialized_values)
    )


def snapshot_distribution_list_visualize_kwargs(kernel_keys, kernel_structure, serialized_values,
                                                num_values_per_param):
    return dict(
        components=[vexpr.web.position_distribution_list_view(class_name="mean", key="mean"),
                    vexpr.web.expression_distribution_list_view(class_name="kernel",
                                                                keys=kernel_keys,
                                                                text=repr(kernel_structure)),
                    vexpr.web.scalar_distribution_list_view(class_name="noise", key="noise")],
        encoded_to_snapshot=vexpr.web.scalar_distribution_list_snapshot_js(headers(kernel_keys),
                                                                           num_values_per_param),
        **snapshot_common_visualize_kwargs(serialized_values)
    )


class MeanNoiseKernelDistributionTimeline:
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
                **timeline_distribution_visualize_kwargs(self.kernel_keys,
                                                         self.kernel_structure,
                                                         self.outfile,
                                                         self.num_values_per_param),
            )
        )


class MeanNoiseKernelDistributionNotebookTimeline:
    def __init__(self, model, num_values_per_param):
        self.outfile = io.StringIO()
        (self.kernel_structure,
         self.kernel_keys,
         initial_values) = aliased_kernel(model)
        print(row(model, precomputed_kernel_values=initial_values),
              file=self.outfile)

        self.element_id = vexpr.notebook.visualize_timeline(
            **timeline_distribution_visualize_kwargs(self.kernel_keys,
                                                     self.kernel_structure,
                                                     self.outfile,
                                                     num_values_per_param))

    def on_update(self, model):
        print(row(model), file=self.outfile)
        vexpr.notebook.update_timeline(self.element_id, self.outfile.getvalue())


class MeanNoiseKernelDistributionSnapshot:
    def __init__(self, model, num_values_per_param):
        (kernel_structure,
         kernel_keys,
         initial_values) = aliased_kernel(model)
        serialized_values = row(model, precomputed_kernel_values=initial_values)
        self.html = vexpr.web.full_html(
            vexpr.web.visualize_snapshot_html(
                **snapshot_distribution_visualize_kwargs(kernel_keys,
                                                         kernel_structure,
                                                         serialized_values,
                                                         num_values_per_param),
            )
        )

    def full_html(self):
        return self.html


class MeanNoiseKernelDistributionNotebookSnapshot:
    def __init__(self, model, num_values_per_param):
        (kernel_structure,
         kernel_keys,
         initial_values) = aliased_kernel(model)
        serialized_values = row(model, precomputed_kernel_values=initial_values)

        self.element_id = vexpr.notebook.visualize_snapshot(
            **snapshot_distribution_visualize_kwargs(kernel_keys, kernel_structure,
                                                     serialized_values,
                                                     num_values_per_param))

    def on_update(self, model):
        serialized_values = row(model)
        vexpr.notebook.update_snapshot(self.element_id, serialized_values)



class MeanNoiseKernelDistributionListSnapshot:
    def __init__(self, models, num_values_per_param):
        (kernel_structure,
         kernel_keys,
         _) = aliased_kernel(models[0])
        serialized_values = "\n".join([row(model) for model in models])
        self.html = vexpr.web.full_html(
            vexpr.web.visualize_snapshot_html(
                **snapshot_distribution_list_visualize_kwargs(kernel_keys,
                                                              kernel_structure,
                                                              serialized_values,
                                                              num_values_per_param),
            )
        )

    def full_html(self):
        return self.html


class SamplingMeanNoiseKernelDistributionListSnapshot:
    def __init__(self, models_lists, samples_per_model=10):
        (kernel_structure,
         kernel_keys,
         _) = aliased_kernel(models_lists[0][0])
        serialized_values = "\n".join([sampling_row(models, samples_per_model)
                                       for models in models_lists])

        num_values_per_param = [samples_per_model * len(models)
                                for models in models_lists]

        self.html = vexpr.web.full_html(
            vexpr.web.visualize_snapshot_html(
                **snapshot_distribution_list_visualize_kwargs(
                    kernel_keys, kernel_structure, serialized_values,
                    num_values_per_param),
            )
        )

    def full_html(self):
        return self.html




class MeanNoiseKernelDistributionNotebookSnapshot:
    def __init__(self, models, num_values_per_param):
        (kernel_structure,
         kernel_keys,
         _) = aliased_kernel(models[0])
        serialized_values = "\n".join([row(model) for model in models])

        self.element_id = vexpr.notebook.visualize_snapshot(
            **snapshot_distribution_list_visualize_kwargs(kernel_keys,
                                                          kernel_structure,
                                                          serialized_values,
                                                          num_values_per_param))

    def on_update(self, model):
        serialized_values = row(model)
        vexpr.notebook.update_snapshot(self.element_id, serialized_values)
