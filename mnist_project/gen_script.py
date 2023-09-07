import json
import os
import pickle


from src.scheduling import (
    Lock, Phase, sbatch_gen_task, sbatch_trial_task, parse_results
)

from src.sweeps import CONFIGS


def go(sweep_name):
    config = CONFIGS[sweep_name]

    project_dir = os.getcwd()
    sweep_dir = os.path.join(project_dir, "results", sweep_name)
    lock_path = os.path.join(sweep_dir, "lock")
    state_pkl_path = os.path.join(sweep_dir, "state.pkl")

    with Lock(lock_path):
        (prev_configs,
         prev_trial_dirs,
         pending_configs) = parse_results(sweep_name)

        with open(state_pkl_path, "rb") as f:
            state = pickle.load(f)

        trial_id = str(len(state["phase_by_trial_id"]))
        trial_dir = os.path.join(sweep_dir, trial_id)
        os.makedirs(trial_dir, exist_ok=True)
        state["phase_by_trial_id"][trial_id] = Phase.BO
        state["is_bo_queued"] = False
        with open(state_pkl_path, "wb") as f:
            pickle.dump(state, f)

    gen_function = config["gen_function"]
    args = gen_function(trial_dir, prev_configs, prev_trial_dirs,
                        pending_configs)

    with open(os.path.join(trial_dir, "args.json"), "w") as f:
        json.dump(args, f)

    with Lock(lock_path):
        with open(state_pkl_path, "rb") as f:
            state = pickle.load(f)

        state["phase_by_trial_id"][trial_id] = Phase.TRIAL
        p1 = sbatch_trial_task(sweep_name, trial_id)

        total_trials = config["gen_function"].total_trials
        if not state["is_bo_queued"] \
           and len(state["phase_by_trial_id"]) < total_trials:
            p2 = sbatch_gen_task(sweep_name)
            state["is_bo_queued"] = True
        else:
            p2 = None

        with open(state_pkl_path, "wb") as f:
            pickle.dump(state, f)

    if p1 is not None:
        p1.wait()
    if p2 is not None:
        p2.wait()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-name", type=str, required=True)
    args = parser.parse_args()

    go(**vars(args))
