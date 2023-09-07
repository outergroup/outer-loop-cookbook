import os
import shutil
import pprint

from src.sweeps import CONFIGS
from src import gen
from src.gp import mnist_metrics


def run(sweep_name):
    config = CONFIGS[sweep_name]

    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)
    sweep_dir = os.path.join(project_dir, "results", sweep_name)

    args = gen.BestActual(
        source_sweep_name=sweep_name,
        metric=mnist_metrics.trial_dir_to_loss_y,
    )()

    print("Running with config:")
    pprint.pprint(args)

    trial_function = config["trial_function"]
    trial_function_kwargs = config.get("trial_function_kwargs", {})
    trial_dir = None
    trial_function(trial_dir, args, verbose=1, **trial_function_kwargs)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-name", type=str, default="mnist1")

    cmd_args = parser.parse_args()
    run(**vars(cmd_args))


if __name__ == "__main__":
    main()
