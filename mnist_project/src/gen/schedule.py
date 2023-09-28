class Schedule:
    def __init__(self, *steps):
        base = 0
        self.steps = []
        for n, instance in steps:
            self.steps.append((base + n, instance))
            base += n
        self.total_trials = base


    def __call__(self, trial_dir, prev_configs, prev_trial_dirs, pending_configs):
        n = len(prev_configs) + len(pending_configs)

        for threshold, instance in self.steps:
            if n < threshold:
                return instance(trial_dir, prev_configs, prev_trial_dirs, pending_configs)

        raise ValueError(f"No schedule item found for trial number {n}")
