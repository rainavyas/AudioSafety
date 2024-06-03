from .whisper import WhisperModel

def load_model(core_args, device=None):
    return WhisperModel(core_args.model_name, device=device, task=core_args.task, language=core_args.language)