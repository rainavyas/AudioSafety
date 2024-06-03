import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import os
from tqdm import tqdm
from whisper.audio import load_audio

from .base_safety_filter_class import BaseSafetyFilter
from src.tools.tools import AverageMeter



class BaseSafetyFilterTrainer(BaseSafetyFilter):
    '''
       Train Safety Filter
    '''
    def __init__(self, safety_args, whisper_model, device, lr=1e-3):
        BaseSafetyFilter.__init__(self, safety_args, whisper_model, device)
        self.optimizer = torch.optim.AdamW(self.safety_filter_model.parameters(), lr=lr, eps=1e-8)

    def _prep_dl(self):
        raise NotImplementedError
    
    def train_step(self):
        raise NotImplementedError

    def train_process(self, train_data, cache_dir):

        fpath = f'{cache_dir}/models'
        if not os.path.isdir(fpath):
            os.mkdir(fpath)

        train_dl = self._prep_dl(train_data, bs=self.safety_args.bs, shuffle=True)

        for epoch in range(self.safety_args.max_epochs):
            # train for one epoch
            print('current lr {:.5e}'.format(self.optimizer.param_groups[0]['lr']))
            self.train_step(train_dl, epoch)

            if epoch==self.safety_args.max_epochs-1 or (epoch+1)%self.safety_args.save_freq==0:
                # save model at this epoch
                if not os.path.isdir(f'{fpath}/epoch{epoch+1}'):
                    os.mkdir(f'{fpath}/epoch{epoch+1}')
                state = self.safety_filter_model.state_dict()
                torch.save(state, f'{fpath}/epoch{epoch+1}/model.th')










            

