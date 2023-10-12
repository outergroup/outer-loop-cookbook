import csv
import os

import numpy as np
import matplotlib.pyplot as plt

results_dir = "results/perf"

all_num_models_values = list(range(1, 21))

# Parse CSV files
means_cpu_naive = []
means_gpu_naive = []
means_tot_naive = []
means_cpu_vectorized = []
means_gpu_vectorized = []
means_tot_vectorized = []
for vectorize in [True, False]:
    vectorize_str = "true" if vectorize else "false"
    for num_models in all_num_models_values:
        filename = f"{results_dir}/mnist1-VexprPartialHandsOnGP-benchmark_covariance_fit-repetitions30-compilefalse-vectorize{vectorize_str}-num_models{num_models}.csv"
        with open(filename) as f:
            reader = csv.reader(f)
            next(reader) # skip header
            duration, cpu_wait_time, gpu_active_time = [float(v)/1e9 for v in next(reader)]
            cpu_active_time = duration - cpu_wait_time
        if vectorize:
            means_cpu_vectorized.append(cpu_active_time)
            means_gpu_vectorized.append(gpu_active_time)
            means_tot_vectorized.append(duration)
        else:
            means_cpu_naive.append(cpu_active_time)
            means_gpu_naive.append(gpu_active_time)
            means_tot_naive.append(duration)

x = np.array(all_num_models_values)

fmt_tot = "-"
fmt_gpu = "--"
fmt_cpu = ":"

plt.figure(figsize=(7, 5))

plt.plot(x, means_tot_naive, fmt_tot, label="Naive Result", color='C0')
plt.plot(x, means_tot_vectorized, fmt_tot, label="Vectorized Result", color='C1')
plt.plot(x, means_gpu_naive, fmt_gpu, label="GPU time (naive)", color='C0', alpha=0.3, linewidth=4)
plt.plot(x, means_gpu_vectorized, fmt_gpu, label="GPU time (vectorized)", color='C1', alpha=0.3, linewidth=4)
plt.plot(x, means_cpu_naive, fmt_cpu, label="CPU time (naive)", color='C0', alpha=0.3, linewidth=4)
plt.plot(x, means_cpu_vectorized, fmt_cpu, label="CPU time (vectorized)", color='C1', alpha=0.3, linewidth=4)

plt.title("Covariance Kernel Benchmark: Fit multiple GPs in parallel")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)
# Set xticks to only show whole numbers
plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.xlabel("# of models trained in parallel")
plt.ylabel("Runtime\n(s)")
plt.tight_layout()

out_dir = "images"
os.makedirs(out_dir, exist_ok=True)
filename = os.path.join("images", "benchmark_fit.svg") 
print(f"Saving figure to {filename}")
plt.savefig(filename)