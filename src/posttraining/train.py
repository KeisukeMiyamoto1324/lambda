import math
import os
from pathlib import Path
import sys

from dotenv import load_dotenv
import torch

# ---------------------------------------------------------
# Add the project root so direct script execution can import
# modules through the src package path.
# ---------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.posttraining.artifacts import save_chat_model
from src.posttraining.cli import parse_args
from src.posttraining.dataloaders import build_dataloaders
from src.posttraining.model_setup import build_tokenizer
from src.posttraining.model_setup import download_base_model
from src.posttraining.model_setup import load_base_model
from src.posttraining.trainer import train_stage
from src.shared.device_utils import resolve_accelerator
from src.shared.device_utils import resolve_device_count
from src.shared.device_utils import resolve_devices
from src.shared.device_utils import resolve_precision
from src.shared.device_utils import resolve_strategy
from src.shared.pytorch_artifacts import push_pytorch_model_artifacts
load_dotenv()


def main() -> None:
    # ---------------------------------------------------------
    # Parse CLI input, prepare output storage, and resolve the
    # active accelerator configuration.
    # ---------------------------------------------------------
    args = parse_args()
    model_dir = Path(args.output_path)
    model_dir.mkdir(parents=True, exist_ok=True)
    accelerator = resolve_accelerator()
    devices = resolve_devices(devices=args.devices)
    device_count = resolve_device_count(accelerator=accelerator, devices=devices)
    strategy = resolve_strategy(accelerator=accelerator, device_count=device_count)
    precision = resolve_precision(accelerator=accelerator)

    # ---------------------------------------------------------
    # Download the base artifacts and build all SFT dataloaders from
    # the shared chat template before configuring the LR schedule.
    # ---------------------------------------------------------
    base_model_dir = download_base_model(base_model_id=args.base_model_id)
    tokenizer = build_tokenizer(base_model_dir=base_model_dir, output_path=model_dir)
    train_dataloader, validation_dataloader, max_steps = build_dataloaders(
        tokenizer=tokenizer,
        max_len=args.max_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        accelerator=accelerator,
        repeat_epochs=args.repeat_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        device_count=device_count,
    )
    min_learning_rate = args.learning_rate * args.min_learning_rate_ratio
    steps_per_epoch = max_steps // args.repeat_epochs
    warmup_steps = math.ceil(steps_per_epoch * args.lr_warmup_epochs)

    # ---------------------------------------------------------
    # Load the base model with posttraining loss and LR settings,
    # then optionally replace its weights for a fresh continued run.
    # ---------------------------------------------------------
    model, model_config = load_base_model(
        base_model_dir=base_model_dir,
        tokenizer=tokenizer,
        learning_rate=args.learning_rate,
        accelerator=accelerator,
        loss_chunk_size=args.loss_chunk_size,
        lr_warmup_steps=warmup_steps,
        lr_total_steps=max_steps,
        min_learning_rate=min_learning_rate,
    )

    if args.continue_from_model:
        model_state = torch.load(
            Path(args.continue_from_model),
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(model_state)

    args.posttraining_steps = max_steps
    args.device_count = device_count
    args.global_batch_size = args.batch_size * device_count
    args.global_effective_batch_size = (
        args.batch_size * args.gradient_accumulation_steps * device_count
    )
    args.min_learning_rate = min_learning_rate
    args.posttraining_warmup_steps = warmup_steps

    # ---------------------------------------------------------
    # Run lambda-chat instruction tuning for the requested number of
    # passes through the train split.
    # ---------------------------------------------------------
    trainer = train_stage(
        model=model,
        model_dir=model_dir,
        stage_name="lambda-chat",
        max_steps=max_steps,
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=precision,
        args=args,
    )

    # ---------------------------------------------------------
    # Save the final model after lambda-chat tuning completes.
    # ---------------------------------------------------------
    if not trainer.is_global_zero:
        return

    save_chat_model(
        model=model,
        model_dir=model_dir,
        model_config=model_config,
        args=args,
        pad_token_id=tokenizer.token_to_id(tokenizer.pad_token),
        bos_token_id=tokenizer.token_to_id(tokenizer.bos_token),
        eos_token_id=tokenizer.token_to_id(tokenizer.eos_token),
        end_of_turn_token_id=tokenizer.token_to_id(tokenizer.end_of_turn_token),
    )

    # ---------------------------------------------------------
    # Optionally publish the completed instruction-tuned artifacts
    # to the configured Hugging Face model repository.
    # ---------------------------------------------------------
    if args.push_to_hub:
        push_pytorch_model_artifacts(
            output_path=model_dir,
            repo_id=os.environ["HF_REPO_IT"],
            private=True,
            commit_message="Upload lambda instruction-tuned model",
        )


if __name__ == "__main__":
    main()
