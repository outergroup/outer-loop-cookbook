import json
import os
import pickle

from src.scheduling import Lock, Phase, sbatch_gen_task

from src.sweeps import CONFIGS


def go(sweep_name, trial_id):
    config = CONFIGS[sweep_name]

    project_dir = os.getcwd()
    sweep_dir = os.path.join(project_dir, "results", sweep_name)
    trial_dir = os.path.join(sweep_dir, trial_id)
    args_filename = os.path.join(trial_dir, "args.json")
    with open(args_filename, "r") as f:
        args = json.load(f)

    # run trial, saving results to out folder as they are acquired
    trial_function = config["trial_function"]
    trial_function_kwargs = config.get("trial_function_kwargs", {})
    trial_function(trial_dir, args, **trial_function_kwargs)

    state_pkl_path = os.path.join(sweep_dir, "state.pkl")
    lock_path = os.path.join(sweep_dir, "lock")
    with Lock(lock_path):
        with open(state_pkl_path, "rb") as f:
            state = pickle.load(f)

        state["phase_by_trial_id"][trial_id] = Phase.DONE

        total_trials = config["gen_function"].total_trials
        if not state["is_bo_queued"] \
           and len(state["phase_by_trial_id"]) < total_trials:
            p = sbatch_gen_task(sweep_name)
            state["is_bo_queued"] = True
        else:
            p = None

        with open(state_pkl_path, "wb") as f:
            pickle.dump(state, f)

    if p is not None:
        p.wait()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-name", type=str, required=True)
    parser.add_argument("--trial-id", type=str, required=True)
    cmd_args = parser.parse_args()

    go(**vars(cmd_args))
