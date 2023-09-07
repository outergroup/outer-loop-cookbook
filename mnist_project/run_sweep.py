import os
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

    p = sbatch_gen_task(sweep_name,
                        target=("slurm" if slurm else "local"))
    if p is not None:
        p.wait()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-name", type=str, required=True)
    parser.add_argument("--slurm", action="store_true")

    cmd_args = parser.parse_args()
    run(**vars(cmd_args))


if __name__ == "__main__":
    main()
