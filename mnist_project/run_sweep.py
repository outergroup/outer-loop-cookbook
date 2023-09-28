import os
import pickle
import shutil


def run(sweep_name, slurm):
    import pickle
    from src.scheduling import Lock, sbatch_gen_task

    # set current working directory to the directory of this file
    # (my slurm commands assume this)
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)

    sweep_dir = os.path.join(project_dir, "results", sweep_name)
    shutil.rmtree(sweep_dir, ignore_errors=True)
    os.makedirs(sweep_dir)

    if slurm:
        # create lock, locking now to verify filesystem supports locking
        lock_path = os.path.join(sweep_dir, "lock")
        with Lock(lock_path):
            state_pkl_path = os.path.join(sweep_dir, "state.pkl")

            state = dict(
                phase_by_trial_id={},
                is_bo_queued=True,
            )

            with open(state_pkl_path, "wb") as f:
                print(f"Creating {state_pkl_path}")
                pickle.dump(state, f)

        force_local = False

        p = sbatch_gen_task(sweep_name,
                            target=("local" if force_local else  "slurm"))
        if p is not None:
            p.wait()
    else:
        # run a loop, generating args then running the trial function, always
        # running in a subprocess so that GPU memory is released after each
        import multiprocessing

        from src.sweeps import CONFIGS
        config = CONFIGS[sweep_name]
        total_trials = config["gen_function"].total_trials

        state = dict(
            phase_by_trial_id={},
        )
        state_pkl_path = os.path.join(sweep_dir, "state.pkl")
        with open(state_pkl_path, "wb") as f:
            print(f"Creating {state_pkl_path}")
            pickle.dump(state, f)

        for i in range(total_trials):
            trial_id = str(i)
            trial_dir = os.path.join(sweep_dir, str(i))
            p = multiprocessing.Process(target=generate_config,
                                        args=(sweep_name, trial_id))
            p.start()
            p.join()
            if p.exitcode != 0:
                break

            p = multiprocessing.Process(target=run_trial,
                                        args=(sweep_name, trial_id))
            p.start()
            p.join()
            if p.exitcode != 0:
                break


def generate_config(sweep_name, trial_id):
    import json
    from src.scheduling import parse_results
    from src.sweeps import CONFIGS
    config = CONFIGS[sweep_name]

    project_dir = os.getcwd()
    sweep_dir = os.path.join(project_dir, "results", sweep_name)

    (prev_configs,
     prev_trial_dirs,
     pending_configs) = parse_results(sweep_name)

    state_pkl_path = os.path.join(sweep_dir, "state.pkl")
    with open(state_pkl_path, "rb") as f:
        state = pickle.load(f)

    trial_dir = os.path.join(sweep_dir, trial_id)
    os.makedirs(trial_dir, exist_ok=True)

    gen_function = config["gen_function"]
    args = gen_function(trial_dir, prev_configs, prev_trial_dirs,
                        pending_configs)

    with open(os.path.join(trial_dir, "args.json"), "w") as f:
        json.dump(args, f)


def run_trial(sweep_name, trial_id):
    import json
    from src.scheduling import Phase
    from src.sweeps import CONFIGS
    config = CONFIGS[sweep_name]

    project_dir = os.getcwd()
    sweep_dir = os.path.join(project_dir, "results", sweep_name)
    trial_dir = os.path.join(sweep_dir, trial_id)
    args_filename = os.path.join(trial_dir, "args.json")
    with open(args_filename, "r") as f:
        args = json.load(f)

    print(args)
    # run trial, saving results to out folder as they are acquired
    trial_function = config["trial_function"]
    trial_function_kwargs = config.get("trial_function_kwargs", {})
    trial_function(trial_dir, args, **trial_function_kwargs)

    state_pkl_path = os.path.join(sweep_dir, "state.pkl")
    with open(state_pkl_path, "rb") as f:
        state = pickle.load(f)
    state["phase_by_trial_id"][trial_id] = Phase.DONE
    with open(state_pkl_path, "wb") as f:
        pickle.dump(state, f)



def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-name", type=str, required=True)
    parser.add_argument("--slurm", action="store_true")

    cmd_args = parser.parse_args()
    run(**vars(cmd_args))


if __name__ == "__main__":
    main()
