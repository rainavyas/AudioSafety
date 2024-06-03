"""
    Evaluate safety filter
"""

import sys
import os
import torch

from src.tools.tools import get_default_device, set_seeds
from src.tools.args import core_args, safety_args
from src.tools.saving import (
    base_path_creator,
    safety_base_path_creator_eval,
    safety_base_path_creator_train,
)
from src.data.load_data import load_data
from src.models.load_model import load_model
from src.safety_filter.selector import select_eval_safety_filter

if __name__ == "__main__":

    # get command line arguments
    core_args, c = core_args()
    safety_args, s = safety_args()

    print(core_args)
    print(safety_args)

    set_seeds(core_args.seed)
    if not safety_args.transfer:
        base_path = base_path_creator(core_args)
        safety_base_path = safety_base_path_creator_eval(safety_args, base_path)
    else:
        base_path = None
        safety_base_path = None

    # Save the command run
    if not os.path.isdir("CMDs"):
        os.mkdir("CMDs")
    with open("CMDs/eval_safety_filter.cmd", "a") as f:
        f.write(" ".join(sys.argv) + "\n")

    # Get the device
    if core_args.force_cpu:
        device = torch.device("cpu")
    else:
        device = get_default_device(core_args.gpu_id)
    print(device)

    # Load the data
    train_data, test_data = load_data(core_args)
    if safety_args.eval_train:
        test_data = train_data

    # Load the model
    model = load_model(core_args, device=device)

    # load safety filter for evaluation
    safety_filter = select_eval_safety_filter(safety_args, core_args, model, device=device)

    # evaluate
    if not safety_args.transfer:
        safety_model_dir = f"{safety_base_path_creator_train(safety_args, base_path)}/models"
    else:
        safety_model_dir = safety_args.safety_model_dir

    # 1) No safety filter
    if not safety_args.not_none:
        print("No safety filter")
        out = safety_filter.eval_safety_filter(
            test_data,
            safety_model_dir=safety_model_dir,
            safety_epoch=-1,
            cache_dir=safety_base_path,
            force_run=safety_args.force_run,
            metrics=safety_args.eval_metrics,
        )
        print(out)
        print()

    # 2) Safety filter
    print("Safety filter")
    out = safety_filter.eval_safety_filter(
        test_data,
        safety_model_dir=safety_model_dir,
        safety_epoch=safety_args.safety_epoch,
        cache_dir=safety_base_path,
        force_run=safety_args.force_run,
        metrics=safety_args.eval_metrics,
    )
    print(out)
    print()
