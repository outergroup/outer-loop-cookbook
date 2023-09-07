from ..scheduling import parse_results


class BestActual:
    def __init__(self, source_sweep_name, metric):
        self.source_sweep_name = source_sweep_name
        self.metric = metric

    def __call__(self, *args, **kwargs):
        (prev_configs,
         prev_trial_dirs,
         _) = parse_results(self.source_sweep_name)

        best_config = None
        best_loss = None
        for config, trial_dir in zip(prev_configs, prev_trial_dirs):
            loss = self.metric(trial_dir)
            if loss is None:
                continue
            loss = loss[0]
            if best_loss is None or loss < best_loss:
                best_loss = loss
                best_config = config

        return best_config
