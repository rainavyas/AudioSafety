import torch
import torch.nn as nn
from whisper.audio import log_mel_spectrogram, pad_or_trim, N_SAMPLES, N_FRAMES, load_audio

class PrependSafetyFilter(nn.Module):
    '''
        Safety Filter where audio segment is prepended to target speech signal
    '''
    def __init__(self, tokenizer, prepend_size=10240, device=None):
        super(PrependSafetyFilter, self).__init__()
        self.prepend_size = prepend_size
        self.tokenizer = tokenizer
        self.device = device

        # self.sot_ids = self.tokenizer.sot_sequence
        self.sot_ids = self.tokenizer.sot_sequence_including_notimestamps
        self.len_sot_ids = len(torch.tensor(self.sot_ids))

        self.prepend_segment = nn.Parameter(torch.rand(prepend_size))

    
    def forward(self, audio_vector, whisper_model, decoder_input=None):
        '''
            audio_vector: Torch.tensor: [Batch x Audio Length]
            whisper_model: encoder-decoder model

            Returns the logits
        '''
        # prepend audio segment
        X = self.prepend_segment.unsqueeze(0).expand(audio_vector.size(0), -1)
        modified_audio_vector = torch.cat((X, audio_vector), dim=1)

        # forward pass through full model
        mel = self._audio_to_mel(modified_audio_vector, whisper_model)
        return self._mel_to_logit(mel, whisper_model, decoder_input=decoder_input)
    

    def _audio_to_mel(self, audio: torch.Tensor, whisper_model):
        '''
            audio: [Batch x Audio length]
            based on https://github.com/openai/whisper/blob/main/whisper/audio.py
        '''
        n_mels = whisper_model.model.dims.n_mels
        padded_mel = log_mel_spectrogram(audio, n_mels, padding=N_SAMPLES)
        mel = pad_or_trim(padded_mel, N_FRAMES)
        return mel
    
    def _mel_to_logit(self, mel: torch.Tensor, whisper_model, decoder_input=None):
        '''
            Forward pass through the whisper model of the mel vectors
            expect mel vectors passed as a batch and padded to 30s of audio length
            mel: torch.Tensor [B x dim x num_vectors]
        '''
        # create batch of start of transcript tokens
        sot_ids = torch.tensor(self.sot_ids)
        sot_ids = sot_ids.to(self.device)
        decoder_input_ids = sot_ids.unsqueeze(0).expand(mel.size(0), -1)
        if decoder_input is not None:
            decoder_input_ids = torch.cat((decoder_input_ids, decoder_input), dim=1)
        return whisper_model.model.forward(mel, decoder_input_ids)
    
    def transcribe(self,
        whisper_model,
        audio,
        apply_safety=True,
        without_timestamps=False
    ):

        '''
            Mimics the original Whisper transcribe functions but prepends the safety audio segment
            in the audio space

                apply_safety parameter is a boolean to apply the safety or not
        '''
        if apply_safety:
            # prepend audio
            if isinstance(audio, str):
                audio = load_audio(audio)
            audio = torch.from_numpy(audio).to(self.device)
            audio = torch.cat((self.prepend_segment, audio), dim=0)

        return whisper_model.predict(audio, without_timestamps=without_timestamps)

