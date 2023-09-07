import glob
import os

import torch


def trial_dir_to_y(trial_dir, num_expected_files=None, min_value=None):
    result_files = glob.glob(os.path.join(trial_dir, f"result-*.pt"))

    if num_expected_files is not None:
        if len(result_files) != num_expected_files:
            print(f"Skipping {trial_dir} because it has {len(result_files)} "
                  f"result files, but expected {num_expected_files}")
            return None

    total_log_density = 0

    for result_file in result_files:
        result = torch.load(result_file, map_location="cpu")
        posterior = result["posterior"]
        Y = result["observed_Y"]
        distribution = posterior.distribution

        log_density = distribution.log_prob(Y.squeeze(-1))

        # log_density is n log densities, one for each cv fold.
        # as an objective, we can simply add all the log densities.
        # we are testing with different n, so this will give more weight to
        # the larger n.
        total_log_density += log_density.sum().item()

    if min_value is not None:
        total_log_density = max(total_log_density, min_value)

    return [total_log_density]