import numpy as np
import pandas as pd
import rows2prose as r2p
import rows2prose.notebook as r2p_nb
import torch
import vexpr as vp
import vexpr.web


def aliased_kernel(model):
    with torch.no_grad():
        parameters = {name: module.value
                      for name, module in model.covar_module.state.items()}

    expr = vp.partial_eval(model.kernel_viz_vexpr, parameters)
    aliased_expr, aliases, values = vexpr.web.alias_values(expr)
    return aliased_expr, aliases, values


snapshot_html = """
<style>
.vexpr-code {{
color: gray;
}}
circle.point {{
fill: blue;
opacity: 0.2;
}}
</style>
<p><strong>Mean:</strong> constant</p>
<pre data-key="mean" class="mean"></pre>
<p><strong>Covariance:</strong> Start with matrix formed by kernel</p>
<pre class="kernel" style="border: 1px solid silver; border-radius: 5px; padding: 10px; overflow: auto; height: 400px;"></pre>
<p><strong>Noise:</strong> Take that matrix and add the following number to each value along the diagonal. <em>(Plotted on log scale.)</em></p>
<pre data-key="noise" class="noise"></pre>"""



timeline_html = """
<style>
.vexpr-code {{
color: gray;
}}
circle.point {{
fill: blue;
opacity: 0.2;
}}
</style>
<div style="width:800px; padding: 10px;">
  <div class="timesteps"></div>
  <p><strong>Mean:</strong> constant</p>
  <pre data-key="mean" class="mean"></pre>
  <p><strong>Covariance:</strong> Start with matrix formed by kernel</p>
  <pre class="kernel" style="border: 1px solid silver; border-radius: 5px; padding: 10px; overflow: auto; height: 400px;"></pre>
  <p><strong>Noise:</strong> Take that matrix and add the following number to each value along the diagonal. <em>(Plotted on log scale.)</em></p>
  <pre data-key="noise" class="noise"></pre>
</div>"""


def snapshot_rows(model):
    (kernel_structure,
     kernel_keys,
     kernel_values) = aliased_kernel(model)
    kernel_keys = ["mean", "noise"] + kernel_keys
    kernel_values = [
        model.mean_module.constant.detach().tolist(),
        model.likelihood.noise.squeeze(-1).detach().tolist(),
    ] + kernel_values
    kernel_values = [np.array(arr, dtype="float32") for arr in kernel_values]
    if len(kernel_values[0].shape) == 0:
        kernel_values = [arr[np.newaxis] for arr in kernel_values]
    return (kernel_structure,
            kernel_keys,
            pd.DataFrame(dict(zip(kernel_keys, kernel_values))))


def timeline_rows(model, timestep):
    (kernel_structure,
     kernel_keys,
     kernel_values) = aliased_kernel(model)
    kernel_keys = ["mean", "noise"] + kernel_keys
    kernel_values = [
        model.mean_module.constant.detach().tolist(),
        model.likelihood.noise.squeeze(-1).detach().tolist(),
    ] + kernel_values
    kernel_values = [np.array(arr, dtype="float32")
                         for arr in kernel_values]
    kernel_keys = kernel_keys + ["i_timestep"]
    n = (kernel_values[0].shape[0]
         if len(kernel_values[0].shape) > 0
         else 1)
    kernel_values = kernel_values + [
        np.repeat(timestep, n).astype("int32")
    ]

    return (kernel_structure,
            kernel_keys,
            pd.DataFrame(dict(zip(kernel_keys, kernel_values))))


class MeanNoiseKernelTimeline:
    def __init__(self, model):
        (self.kernel_structure,
         self.kernel_keys,
         self.df) = timeline_rows(model, 0)

    def on_update(self, model, timestep):
        (_, _, df) = timeline_rows(model, timestep)
        self.df = pd.concat([self.df, df], ignore_index=True)

    def full_html(self):
        viz = r2p.Timeline
        return r2p.full_html(
            r2p.static(self.df, timeline_html,
                       viz(viz.time_control(class_name="timesteps"),
                           viz.position_view(class_name="mean"),
                           viz.expression_view(
                               class_name="kernel",
                               keys=self.kernel_keys,
                               text=repr(self.kernel_structure)),
                           viz.positive_scalar_view(class_name="noise")))
        )


class MeanNoiseKernelNotebookSnapshot:
    def __init__(self, model):
        (kernel_structure,
         kernel_keys,
         df) = snapshot_rows(model)
        viz = r2p.Snapshot
        self.update = r2p_nb.display_dynamic(
            snapshot_html,
            viz(viz.position_view(class_name="mean"),
                viz.expression_view(class_name="kernel", keys=kernel_keys,
                                    text=repr(kernel_structure)),
                viz.positive_scalar_view(class_name="noise"))
        )
        self.update(df)

    def on_update(self, model):
        (_, _, df) = snapshot_rows(model)
        self.update(df)


class MeanNoiseKernelNotebookTimeline:
    def __init__(self, model):
        (kernel_structure,
         kernel_keys,
         self.df) = timeline_rows(model, 0)

        viz = r2p.Timeline
        self.update = r2p_nb.display_dynamic(
            timeline_html,
            viz(viz.time_control(class_name="timesteps"),
                viz.position_view(class_name="mean"),
                viz.expression_view(class_name="kernel", keys=kernel_keys,
                                    text=repr(kernel_structure)),
                viz.positive_scalar_view(class_name="noise")))
        self.update(self.df)

    def on_update(self, model, timestep):
        (_, _, df) = timeline_rows(model, timestep)
        self.df = pd.concat([self.df, df], ignore_index=True)
        self.update(self.df)


class MeanNoiseKernelDistributionTimeline:
    def __init__(self, model):
        (self.kernel_structure,
         self.kernel_keys,
         self.df) = timeline_rows(model, 0)

    def on_update(self, model, timestep):
        (_, _, df) = timeline_rows(model, timestep)
        self.df = pd.concat([self.df, df], ignore_index=True)

    def full_html(self):
        viz = r2p.DistributionTimeline
        return r2p.full_html(
            r2p.static(
                self.df,
                timeline_html,
                viz(viz.time_control(class_name="timesteps"),
                    viz.scalar_view(class_name="mean"),
                    viz.expression_view(
                        class_name="kernel",
                        keys=self.kernel_keys,
                        text=repr(self.kernel_structure)),
                    viz.scalar_view(class_name="noise"))))


class MeanNoiseKernelDistributionNotebookTimeline:
    def __init__(self, model, num_values_per_param):
        (kernel_structure,
         kernel_keys,
         self.df) = timeline_rows(model, 0)

        viz = r2p.DistributionTimeline
        self.update = r2p_nb.display_dynamic(
            timeline_html,
            viz(viz.time_control(class_name="timesteps"),
                viz.scalar_view(class_name="mean"),
                viz.expression_view(
                    class_name="kernel",
                    keys=kernel_keys,
                    text=repr(kernel_structure)),
                viz.scalar_view(class_name="noise")))
        self.update(self.df)

    def on_update(self, model):
        (_, _, df) = snapshot_rows(model)
        self.df = pd.concat([self.df, df], ignore_index=True)
        self.update(self.df)


class MeanNoiseKernelDistributionSnapshot:
    def __init__(self, model, num_values_per_param):
        (kernel_structure,
         kernel_keys,
         df) = snapshot_rows(model)

        viz = r2p.DistributionSnapshot
        self.html = r2p.full_html(
            r2p.static(
                df,
                snapshot_html,
                viz(viz.time_control(class_name="timesteps"),
                    viz.scalar_view(class_name="mean"),
                    viz.expression_view(
                        class_name="kernel",
                        keys=kernel_keys,
                        text=repr(kernel_structure)),
                    viz.scalar_view(class_name="noise"))))

    def full_html(self):
        return self.html


class MeanNoiseKernelDistributionNotebookSnapshot:
    def __init__(self, model, num_values_per_param):
        (kernel_structure,
         kernel_keys,
         self.df) = snapshot_rows(model)

        viz = r2p.DistributionSnapshot

        self.update = r2p_nb.display_dynamic(
            timeline_html,
            viz(viz.scalar_view(class_name="mean"),
                viz.expression_view(
                    class_name="kernel",
                    keys=kernel_keys,
                    text=repr(kernel_structure)),
                viz.scalar_view(class_name="noise")))
        self.update(self.df)

    def on_update(self, model):
        (_, _, df) = snapshot_rows(model)
        self.df = pd.concat([self.df, df], ignore_index=True)
        self.update(self.df)



class MeanNoiseKernelDistributionListSnapshot:
    def __init__(self, models, num_values_per_param):
        (kernel_structure,
         kernel_keys,
         df) = snapshot_rows(models)

        viz = r2p.DistributionListSnapshot

        self.html = r2p.full_html(
            r2p.static(df, snapshot_html,
                       viz(viz.scalar_view(class_name="mean"),
                           viz.expression_view(
                               class_name="kernel",
                               keys=kernel_keys,
                               text=repr(kernel_structure)),
                           viz.scalar_view(class_name="noise"))))

    def full_html(self):
        return self.html


class SamplingMeanNoiseKernelDistributionListSnapshot:
    def __init__(self, models_lists, samples_per_model=None):
        dfs = []
        for i, models in enumerate(models_lists):
            for model in models:
                (kernel_structure,
                 kernel_keys,
                 df) = snapshot_rows(model)
                if samples_per_model is not None:
                    df = df.iloc[:samples_per_model]
                df["i_config"] = np.array(i, dtype="int32")
                dfs.append(df)

        # Store on self so that callers can access it.
        self.df = pd.concat(dfs, ignore_index=True)

        viz = r2p.DistributionListSnapshot

        self.html = r2p.full_html(
            r2p.static(self.df, snapshot_html,
                       viz(viz.scalar_view(class_name="mean"),
                           viz.expression_view(
                               class_name="kernel",
                               keys=kernel_keys,
                               text=repr(kernel_structure)),
                           viz.scalar_view(class_name="noise"))))
    def full_html(self):
        return self.html
