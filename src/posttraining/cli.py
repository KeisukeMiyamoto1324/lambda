import argparse

from src.posttraining.model_setup import DEFAULT_BASE_MODEL_ID
from src.shared.cli import require
from src.shared.device_utils import resolve_devices


def parse_args() -> argparse.Namespace:
    # ---------------------------------------------------------
    # Define CLI arguments for lambda-chat SFT from a pretrained base
    # model into a chat-oriented model artifact.
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-id", type=str, default=DEFAULT_BASE_MODEL_ID)
    parser.add_argument("--output-path", type=str, default="models/lambda-1-160m-it")
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--repeat-epochs", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-batches", type=int, default=8)
    parser.add_argument("--val-check-interval", type=int, default=500)
    parser.add_argument("--checkpoint-every-n-steps", type=int, default=1000)
    parser.add_argument("--metric-log-every-n-steps", type=int, default=50)
    parser.add_argument("--devices", type=str, default="auto")
    args = parser.parse_args()

    # ---------------------------------------------------------
    # Validate posttraining-specific runtime values before loading
    # the base model or materializing the remote chat dataset.
    # ---------------------------------------------------------
    try:
        require(args.max_len > 0, "--max-len must be greater than 0")
        require(args.learning_rate > 0.0, "--learning-rate must be greater than 0")
        require(args.batch_size > 0, "--batch-size must be greater than 0")
        require(args.repeat_epochs > 0, "--repeat-epochs must be greater than 0")
        require(args.num_workers >= 0, "--num-workers must be greater than or equal to 0")
        require(args.val_batches > 0, "--val-batches must be greater than 0")
        require(args.val_check_interval > 0, "--val-check-interval must be greater than 0")
        require(args.checkpoint_every_n_steps > 0, "--checkpoint-every-n-steps must be greater than 0")
        require(args.metric_log_every_n_steps > 0, "--metric-log-every-n-steps must be greater than 0")
        resolve_devices(devices=args.devices)
    except ValueError as error:
        parser.error(str(error))

    return args
