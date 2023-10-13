import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

results_dir = "results/perf"

all_num_models_values = list(range(1, 21))

# Parse CSV files
tot = defaultdict(list)
cpu = defaultdict(list)
gpu = defaultdict(list)

for compile in [True, False]:
    compile_str = "true" if compile else "false"
    for vectorize in [True, False]:
        vectorize_str = "true" if vectorize else "false"
        for num_models in all_num_models_values:
            filename = f"{results_dir}/mnist1-VexprPartialHandsOnGP-benchmark_covariance_fit-repetitions30-compile{compile_str}-vectorize{vectorize_str}-num_models{num_models}.csv"
            with open(filename) as f:
                reader = csv.reader(f)
                next(reader) # skip header
                duration, cpu_wait_time, gpu_active_time = [float(v)/1e9 for v in next(reader)]
                cpu_active_time = duration - cpu_wait_time

                sel = (compile, vectorize)
                cpu[sel].append(cpu_active_time)
                gpu[sel].append(gpu_active_time)
                tot[sel].append(duration)

# call to create two side-by-side plots with a shared y axis
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 5))
x = np.array(all_num_models_values)

for compile, ax in zip([False, True], [ax1, ax2]):
    fmt_tot = "-"
    fmt_gpu = "--"
    fmt_cpu = ":"

    ax.plot(x, tot[(compile, False)], fmt_tot, label="Naive Result", color='C0')
    ax.plot(x, tot[(compile, True)], fmt_tot, label="Vectorized Result", color='C1')
    ax.plot(x, gpu[(compile, False)], fmt_gpu, label="GPU time (naive)", color='C0', alpha=0.3, linewidth=4)
    ax.plot(x, gpu[(compile, True)], fmt_gpu, label="GPU time (vectorized)", color='C1', alpha=0.3, linewidth=4)
    ax.plot(x, cpu[(compile, False)], fmt_cpu, label="CPU time (naive)", color='C0', alpha=0.3, linewidth=4)
    ax.plot(x, cpu[(compile, True)], fmt_cpu, label="CPU time (vectorized)", color='C1', alpha=0.3, linewidth=4)

    title = "After torch.compile" if compile else "Before torch.compile"
    ax.set_title(title)
    ax.set_xticks(np.arange(min(x) + 1, max(x)+1, 2.0))
    ax.set_xlabel("# of models trained in parallel")

# set title for the entire plot, independent of the individual titles
fig.suptitle("Covariance Kernel Benchmark: Fit multiple GPs in parallel")

ax1.set_ylabel("Runtime\n(s)")
# edit the legend to contain only the first six entries
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles[:6], labels[:6], loc='upper center', bbox_to_anchor=(0.5, 0.11), ncol=3)
plt.tight_layout()

# set the outside of the figure to be lower, so that we can see the legend
fig.subplots_adjust(bottom=0.24)

out_dir = "images"
os.makedirs(out_dir, exist_ok=True)
filename = os.path.join("images", f"benchmark_fit.svg")
print(f"Saving figure to {filename}")
plt.savefig(filename)
