import enum
import fcntl
import json
import os
import pickle
import subprocess
import time


class Lock:
    def __init__(self, filename):
        self.filename = filename

    def __enter__(self):
        start_time = time.time()
        timeout = 30.0
        f = os.open(self.filename, os.O_RDWR | os.O_CREAT | os.O_TRUNC)
        while time.time() < start_time + timeout:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except (IOError, OSError):
                pass
            time.sleep(1.0)

        self.f = f

    def __exit__(self, exc_type, exc_val, exc_tb):
        fcntl.flock(self.f, fcntl.LOCK_UN)
        os.close(self.f)


def sbatch_gen_task(sweep_name, target=None):
    project_dir = os.getcwd()
    sweep_dir = os.path.join(project_dir, "results", sweep_name)

    if target is None:
        target = ("slurm" if "SLURM_JOB_ID" in os.environ else "local")

    if target == "local":
        return subprocess.Popen(["python", "gen_script.py", "--sweep-name", sweep_name])
    else:
        if "SLURM_JOB_ID" in os.environ:
            # Wait a short time for the current job to finish so that
            # Slurm is free to schedule the next job on the current node.
            job_id = os.environ["SLURM_JOB_ID"]
            extra = f"--dependency=afterok:{job_id}"
        else:
            extra = ""

        out_file = os.path.join(sweep_dir, "slurm-out-%A.txt")
        err_file = os.path.join(sweep_dir, "slurm-err-%A.txt")

        command = f"""
        sbatch -n 1 -G 1 --mem 0 -p t4 -o {out_file} -e {err_file} --requeue --nice {extra} --wrap "python gen_script.py --sweep-name {sweep_name}"
        """
        print(f"Running command: {command}")
        os.system(command)


def sbatch_trial_task(sweep_name, trial_id, target=None):
    project_dir = os.getcwd()
    sweep_dir = os.path.join(project_dir, "results", sweep_name)
    out_dir = os.path.join(sweep_dir, trial_id)
    os.makedirs(out_dir, exist_ok=True)

    if target is None:
        target = ("slurm" if "SLURM_JOB_ID" in os.environ else "local")

    if target == "local":
        return subprocess.Popen(["python", "trial_script.py", "--sweep-name", sweep_name, "--trial-id", str(trial_id)])
    else:
        if "SLURM_JOB_ID" in os.environ:
            # Wait a short time for the current job to finish so that
            # Slurm is free to schedule the next job on the current node.
            job_id = os.environ["SLURM_JOB_ID"]
            extra = f"--dependency=afterok:{job_id}"
        else:
            extra = ""
        out_file = os.path.join(out_dir, "slurm-out-%A.txt")
        err_file = os.path.join(out_dir, "slurm-err-%A.txt")

        command = f"""
        sbatch -n 1 -G 1 --mem 0 -p t4 -o {out_file} -e {err_file} --requeue {extra} --wrap "python trial_script.py --sweep-name {sweep_name} --trial-id {trial_id}"
        """
        print(f"Running command: {command}")
        os.system(command)


Phase = enum.Enum("Phase", ["BO", "TRIAL", "DONE"])


def parse_results(sweep_name):
    sweep_dir = os.path.join("results", sweep_name)
    state_pkl_path = os.path.join(sweep_dir, "state.pkl")
    with open(state_pkl_path, "rb") as f:
        state = pickle.load(f)

    prev_configs = []
    prev_trial_dirs = []
    pending_configs = []
    for prev_trial_id, phase in state["phase_by_trial_id"].items():
        prev_trial_dir = os.path.join(sweep_dir, prev_trial_id)
        if phase.value >= Phase.TRIAL.value:
            with open(os.path.join(prev_trial_dir, "args.json"), "r") as f:
                args = json.load(f)
            if phase.value < Phase.DONE.value:
                pending_configs.append(args)
            else:
                prev_configs.append(args)
                prev_trial_dirs.append(prev_trial_dir)

    return prev_configs, prev_trial_dirs, pending_configs
