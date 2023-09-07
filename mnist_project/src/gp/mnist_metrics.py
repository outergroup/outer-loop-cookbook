import json
import math
import os

import torch


def trial_dir_to_loss_y(trial_dir, log_output=True):
    result_path = os.path.join(trial_dir, "result.json")
    with open(result_path, "r") as f:
        result = json.load(f)

    if "loss" not in result:
        return None

    ret = result["loss"]
    if log_output:
        ret = math.log(ret)

    return [ret]
