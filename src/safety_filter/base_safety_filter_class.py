import json
import os
import torch
from tqdm import tqdm

from src.tools.metrics import eval_wer
from .prepend.model import PrependSafetyFilter

class BaseSafetyFilter():
    '''
        Base class for Safety Filter
    '''
    def __init__(self, safety_args, model, device):
        self.safety_args = safety_args
        self.whisper_model = model
        self.device = device
        self._select_safety_filter_model()

    def _select_safety_filter_model(self):
        if self.safety_args.safety_method == 'prepend':
            self.safety_filter_model = PrependSafetyFilter(self.whisper_model.tokenizer, prepend_size=self.safety_args.prepend_size, device=self.device).to(self.device) 

    def evaluate_metrics(self, hyps, refs, metrics):
        results = {}
        if 'wer' in metrics:
            results['WER'] = eval_wer(hyps, refs)
        return results

    def eval_safety_filter(self, data, safety_model_dir=None, safety_epoch=-1, cache_dir=None, force_run=False, metrics=['wer']):
        '''
            Generates transcriptions with safety filter (saves to cache)
            Computes the metrics specified
                wer: Word Error Rate

            safety_model_dir is the directory with the saved safety_filter model checkpoints
            safey_epoch indicates the checkpoint of the trained safety filter model
                -1 indicates that no-safety filter should be evaluated
        '''
        # check for cache
        fpath = f'{cache_dir}/epoch-{safety_epoch}_predictions.json'
        if os.path.isfile(fpath) and not force_run:
            with open(fpath, 'r') as f:
                hyps = json.load(f)
            refs = [d['ref'] for d in data]
            return self.evaluate_metrics(hyps, refs, metrics)
        
        # no cache
        if safety_epoch == -1:
            apply_safety = False
        else:
            # load safety filter model -- note if epoch=0, that is a rand initialized safety filter
            apply_safety = True
            if safety_epoch > 0:
                self.safety_filter_model.load_state_dict(torch.load(f'{safety_model_dir}/epoch{safety_epoch}/model.th'))

        hyps = []
        for sample in tqdm(data):
            with torch.no_grad():
                hyp = self.safety_filter_model.transcribe(self.whisper_model, sample['audio'], apply_safety=apply_safety)
            hyps.append(hyp)

        refs = [d['ref'] for d in data]
        out = self.evaluate_metrics(hyps, refs, metrics)

        if cache_dir is not None:
            with open(fpath, 'w') as f:
                json.dump(hyps, f)

        return out