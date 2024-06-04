import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from whisper.audio import load_audio

from src.safety_filter.base_trainer import BaseSafetyFilterTrainer
from src.tools.tools import AverageMeter



class PrependSafetyFilterTrainer(BaseSafetyFilterTrainer):
    '''
       Prepend safety filter training
    '''
    def __init__(self, safety_args, whisper_model, device, lr=1e-3):
        BaseSafetyFilterTrainer.__init__(self, safety_args, whisper_model, device, lr=lr)
        self.max_length = 400

    def _loss(self, input_ids, logits, seq_len):
        '''
        Teacher forced cross-entropy loss for Transformer decoder

        input_ids: torch.Tensor: [batch x max_len_sequence]
            Input ids to decoder (padded with zeros)

        seq_len: torch.Tensor: [batch]
            Length of each sequence in the batch for input_ids
        
        logits: torch.Tensor: [batch x (self.safety_filter_model.len_sot_ids + max_len_sequence) x vocab_size]
            Predicted logits from Transformer decoder with (sot_ids, input_ids) at the input of the decoder 
        
        Assume that input_ids and seq_len do not account for the starting tokens, the length of which can be given by
            self.safety_filter_model.len_sot_ids

        '''
        # Get the length of the starting tokens
        len_sot_ids = self.safety_filter_model.len_sot_ids
        
        # Shift logits to match the input_ids
        shifted_logits = logits[:, len_sot_ids-1:-1, :]

        # Flatten logits and targets for cross-entropy
        batch_size, max_len_sequence = input_ids.size()
        vocab_size = logits.size(-1)

        # Create a mask based on sequence lengths
        mask = torch.arange(max_len_sequence, device=self.device).expand(batch_size, max_len_sequence) < seq_len.unsqueeze(1)
        
        # Flatten the mask
        mask = mask.view(-1)
        
        # Compute the loss
        loss = F.cross_entropy(
            shifted_logits.reshape(-1, vocab_size),
            input_ids.reshape(-1),
            reduction='none'
        )
        
        # Apply the mask
        loss = loss * mask
        
        # Sum the losses and divide by the number of non-padded tokens
        loss = loss.sum() / mask.sum().float()
        return loss


    def train_step(self, train_loader, epoch, print_freq=25):
        '''
            Run one train epoch
        '''
        losses = AverageMeter()

        # switch to train mode
        self.safety_filter_model.train()

        for i, (audio, decoder_input, seq_len) in enumerate(train_loader):
            audio = audio.to(self.device)
            decoder_input = decoder_input.to(self.device)
            seq_len = seq_len.to(self.device)

            # Forward pass
            logits = self.safety_filter_model(audio, self.whisper_model, decoder_input=decoder_input)
            loss = self._loss(decoder_input, logits, seq_len)

            # Backward pass and update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.safety_args.clip_val != -1:
                max_val = self.safety_args.clip_val
            else:
                max_val = 100000
            with torch.no_grad():  
                self.safety_filter_model.prepend_segment.clamp_(min=-1*max_val, max=max_val)
        
            # record loss
            losses.update(loss.item(), audio.size(0))
            if i % print_freq == 0:
                print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\tLoss {losses.val:.5f} ({losses.avg:.5f})')
    
    def _pad_sequence(self, tensors, padding_value=0):
        max_length = max(len(tensor) for tensor in tensors)
        padded_tensors = []
        for tensor in tensors:
            padded_tensor = torch.nn.functional.pad(tensor, (0, max_length - len(tensor)), value=padding_value)
            padded_tensors.append(padded_tensor)
        return padded_tensors

    def _prep_dl(self, data, bs=4, shuffle=False):
        '''
        Create batches of audio vectors, token IDs, and text lengths
        '''
        
        print('Loading and batching audio files and ref token IDs')
        audio_vectors = []
        texts = []
        
        print('audio loading')
        for d in tqdm(data):
            audio_np = load_audio(d['audio'])
            audio_vector = torch.from_numpy(audio_np)
            audio_vectors.append(audio_vector)
            texts.append(d['ref'])
        
        audio_vectors = self._pad_sequence(audio_vectors)
        audio_vectors = torch.stack(audio_vectors, dim=0)
        
        # Tokenize texts manually, ensuring padding and truncation
        tokenized_texts = []
        text_lengths = []
        eot_token_id = self.whisper_model.tokenizer.eot
        print('text tokenization')
        
        # Find the length of the longest sequence
        max_seq_len = 0
        for text in texts:
            token_ids = self.whisper_model.tokenizer.encode(text)
            max_seq_len = max(max_seq_len, len(token_ids) + 1)  # +1 for EOT token
        
        # Determine the padding length
        pad_length = min(max_seq_len, self.max_length)
        
        for text in tqdm(texts):
            token_ids = self.whisper_model.tokenizer.encode(text)
            if len(token_ids) >= pad_length:
                token_ids = token_ids[:pad_length-1]  # Truncate to fit EOT token
            token_ids.append(eot_token_id)
            text_lengths.append(len(token_ids))  # Original length before padding
            if len(token_ids) < pad_length:
                token_ids.extend([0] * (pad_length - len(token_ids)))  # Pad
            tokenized_texts.append(torch.tensor(token_ids))

        text_token_ids = torch.stack(tokenized_texts, dim=0)
        text_lengths = torch.tensor(text_lengths)
        
        ds = TensorDataset(audio_vectors, text_token_ids, text_lengths)
        dl = DataLoader(ds, batch_size=bs, shuffle=shuffle)
        return dl












            

