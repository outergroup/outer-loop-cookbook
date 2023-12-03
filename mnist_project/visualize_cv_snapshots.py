import random
from functools import partial

import torch
from botorch.cross_validation import gen_loo_cv_folds

from src.gp import MODELS
from src.gp.gp_utils import (
    configs_dirs_to_X_Y,
)
from src.gp.mnist_metrics import trial_dir_to_loss_y
from src.scheduling import parse_results
from src.sweeps import CONFIGS


def create_cv_model(sweep_name, model_name, vectorize, n_cv, shuffle_seed=None):
    torch.set_default_dtype(torch.float64)

    config = CONFIGS[sweep_name]
    model_cls = MODELS[model_name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    search_space = config["search_space"]
    search_xform = config["search_xform"].to(device)

    model_cls = partial(model_cls,
                        search_space=search_space,
                        search_xform=search_xform,
                        round_inputs=False,
                        vectorize=vectorize,
                        torch_compile=compile)

    configs, trial_dirs, _ = parse_results(sweep_name)
    if shuffle_seed is not None:
        shuffled = list(zip(configs, trial_dirs))
        random.Random(shuffle_seed).shuffle(shuffled)
        configs, trial_dirs = zip(*shuffled)

    X, Y = configs_dirs_to_X_Y(configs, trial_dirs, trial_dir_to_loss_y,
                                config["parameter_space"],
                                search_xform,
                                device=device)

    cv_folds = gen_loo_cv_folds(train_X=X[:n_cv], train_Y=Y[:n_cv])
    return model_cls(cv_folds.train_X, cv_folds.train_Y)


from src.visuals import MeanNoiseKernelDistributionListSnapshot
models = []
num_values_per_param = []
for n_cv in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]:
    model = create_cv_model("mnist1", "VexprFullyJointVisualizedGP", vectorize=True, n_cv=n_cv)
    model.load_state_dict(torch.load(f"results/cv/cv-mnist1-VexprFullyJointVisualizedGP-{n_cv}.pt")["state_dict"])
    models.append(model)
    num_values_per_param.append(n_cv)

visual = MeanNoiseKernelDistributionListSnapshot(models, num_values_per_param)
filename = f"cv_distribution_all.html"
with open(filename, "w") as fout:
    print(f"Writing {filename}")
    fout.write(visual.full_html())
