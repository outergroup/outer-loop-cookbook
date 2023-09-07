class Constant:
    def __init__(self, args):
        self.args = args

    def __call__(self, trial_dir, prev_trials, pending_configs):
        return self.args
