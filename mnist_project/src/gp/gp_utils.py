import outerloop as ol
import torch


def configs_dirs_to_X_Y(configs, trial_dirs, metric, space, xform, 
                        device=None):
    kept_args = []
    kept_y = []
    for args, trial_dir in zip(configs, trial_dirs):
        y = metric(trial_dir)
        if y is not None:
            kept_args.append(args)
            kept_y.append(y)

    X = ol.configs_to_X(space, kept_args,
                        xform=xform.transform, device=device)
    Y = torch.tensor(kept_y, device=device)

    return X, Y