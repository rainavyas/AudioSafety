import argparse


def core_args():
    commandLineParser = argparse.ArgumentParser(allow_abbrev=False)
    commandLineParser.add_argument(
        "--model_name",
        type=str,
        default="whisper-tiny.en",
        help="ASR model",
    )
    commandLineParser.add_argument(
        "--task",
        type=str,
        default="transcribe",
        choices=["transcribe", "translate"],
        help="Whisper task. N.b. translate is only X-en",
    )
    commandLineParser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Source audio language or if performing machine translation do something like fr_en",
    )
    commandLineParser.add_argument(
        "--gpu_id", type=int, default=0, help="select specific gpu"
    )
    commandLineParser.add_argument(
        "--data_name",
        type=str,
        default="librispeech",
        help="dataset for exps; for flores: flores-english-french",
    )
    commandLineParser.add_argument("--seed", type=int, default=1, help="select seed")
    commandLineParser.add_argument(
        "--force_cpu", action="store_true", help="force cpu use"
    )
    return commandLineParser.parse_known_args()


def safety_args():
    # safety filter args
    commandLineParser = argparse.ArgumentParser(allow_abbrev=False)
    # train args
    commandLineParser.add_argument(
        "--safety_method",
        type=str,
        default="prepend",
        choices=["prepend"],
        help="Method for safety filter",
    )
    commandLineParser.add_argument(
        "--max_epochs", type=int, default=20, help="Training epochs for safety filter"
    )
    commandLineParser.add_argument(
        "--save_freq", type=int, default=1, help="Epoch frequency for saving safety filter training"
    )
    commandLineParser.add_argument(
        "--prepend_size", type=int, default=10240, help="Length of prepend audio segment"
    )
    commandLineParser.add_argument(
        "--bs", type=int, default=8, help="Batch size for training safety filter"
    )
    commandLineParser.add_argument(
        "--lr", type=float, default=1e-3, help="Safety filter training learning rate"
    )
    commandLineParser.add_argument(
        "--clip_val",
        type=float,
        default=-1,
        help="Value (maximum) to clip the safety filtr modification. -1 means no clipping",
    )

    # eval safety args
    commandLineParser.add_argument(
        "--safety_epoch",
        type=int,
        default=-1,
        help="Specify which training epoch of safety filter to evaluate; -1 means no safety filter",
    )
    commandLineParser.add_argument(
        "--force_run",
        action="store_true",
        help="Do not load from cache",
    )
    commandLineParser.add_argument(
        "--not_none", action="store_true", help="Do not evaluate the no safety filter setting"
    )
    commandLineParser.add_argument(
        "--eval_train", action="store_true", help="Evaluate safety filter on the train split"
    )
    commandLineParser.add_argument(
        "--eval_metrics",
        type=str,
        default="wer",
        nargs="+",
        help="Which metrics to evaluate from: wer, mute_success",
    )

    # eval safety args for safety transferability
    commandLineParser.add_argument(
        "--transfer",
        action="store_true",
        help="Indicate it is a transferability setting (across model or dataset) for trained safety filter",
    )
    commandLineParser.add_argument(
        "--safety_model_dir",
        type=str,
        default="",
        help="path to trained safety filter to evaluate",
    )
    return commandLineParser.parse_known_args()