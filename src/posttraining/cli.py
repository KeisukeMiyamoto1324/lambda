import argparse
import os
from pathlib import Path

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
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lr-warmup-steps", type=int, default=200)
    parser.add_argument("--min-learning-rate-ratio", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-batches", type=int, default=8)
    parser.add_argument("--validation-cache-path", type=str, default="")
    parser.add_argument("--val-check-interval", type=int, default=1000)
    parser.add_argument("--checkpoint-every-n-steps", type=int, default=2000)
    parser.add_argument("--metric-log-every-n-steps", type=int, default=500)
    parser.add_argument("--loss-chunk-size", type=int, default=32)
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--push-to-hub", action="store_true")

    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume-from-checkpoint", type=str, default="")
    resume_group.add_argument("--continue-from-model", type=str, default="")

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Validate posttraining-specific runtime values before loading
    # the base model or materializing the remote chat dataset.
    # ---------------------------------------------------------
    try:
        require(args.max_len > 0, "--max-len must be greater than 0")
        require(args.learning_rate > 0.0, "--learning-rate must be greater than 0")
        require(
            0 <= args.lr_warmup_steps < args.max_steps,
            "--lr-warmup-steps must be greater than or equal to 0 and less than --max-steps",
        )
        require(
            0.0 <= args.min_learning_rate_ratio <= 1.0,
            "--min-learning-rate-ratio must be between 0.0 and 1.0",
        )
        require(args.batch_size > 0, "--batch-size must be greater than 0")
        require(
            args.gradient_accumulation_steps >= 1,
            "--gradient-accumulation-steps must be greater than or equal to 1",
        )
        require(args.max_steps > 0, "--max-steps must be greater than 0")
        require(args.num_workers >= 0, "--num-workers must be greater than or equal to 0")
        require(args.val_batches > 0, "--val-batches must be greater than 0")
        require(args.val_check_interval > 0, "--val-check-interval must be greater than 0")
        require(args.checkpoint_every_n_steps > 0, "--checkpoint-every-n-steps must be greater than 0")
        require(args.metric_log_every_n_steps > 0, "--metric-log-every-n-steps must be greater than 0")
        require(args.loss_chunk_size > 0, "--loss-chunk-size must be greater than 0")
        resolve_devices(devices=args.devices)
    except ValueError as error:
        parser.error(str(error))

    if args.resume_from_checkpoint and not Path(args.resume_from_checkpoint).is_file():
        parser.error("--resume-from-checkpoint must point to an existing checkpoint file")

    if args.continue_from_model and not Path(args.continue_from_model).is_file():
        parser.error("--continue-from-model must point to an existing model state file")

    if args.push_to_hub and not os.environ.get("HF_REPO_IT"):
        parser.error("HF_REPO_IT is required in the environment when --push-to-hub is set")

    return args
