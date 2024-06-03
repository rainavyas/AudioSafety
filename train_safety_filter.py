"""
Train safety filter segment
"""

import sys
import os
import torch

from src.tools.args import core_args, safety_args
from src.data.load_data import load_data
from src.models.load_model import load_model
from src.tools.tools import get_default_device, set_seeds
from src.safety_filter.selector import select_train_safety_filter
from src.tools.saving import base_path_creator, safety_base_path_creator_train

if __name__ == "__main__":

    # get command line arguments
    core_args, c = core_args()
    safety_args, s = safety_args()

    # set seeds
    set_seeds(core_args.seed)
    base_path = base_path_creator(core_args)
    safety_base_path = safety_base_path_creator_train(safety_args, base_path)

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train_safety_filter.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')

    # Get the device
    if core_args.force_cpu:
        device = torch.device('cpu')
    else:
        device = get_default_device(core_args.gpu_id)
    print(device)

    # load training data
    data, _ = load_data(core_args)

    # load model
    model = load_model(core_args, device=device)

    safety_filter = select_train_safety_filter(safety_args, core_args, model, device=device)
    safety_filter.train_process(data, safety_base_path)
