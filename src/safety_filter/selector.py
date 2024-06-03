from .base_safety_filter_class import BaseSafetyFilter
from .prepend.trainer import PrependSafetyFilterTrainer

def select_eval_safety_filter(safety_args, core_args, model, device=None):
    return BaseSafetyFilter(safety_args, model, device)


def select_train_safety_filter(safety_args, core_args, model, device=None):
    if safety_args.safety_method == 'prepend':
        return PrependSafetyFilterTrainer(safety_args, model, device, lr=safety_args.lr)