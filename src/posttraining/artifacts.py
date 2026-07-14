import argparse
import json
from pathlib import Path

import torch

from src.posttraining.dataset import LAMBDA_CHAT_DATASET_PATH
from src.posttraining.dataset import LAMBDA_CHAT_TRAIN_SPLIT
from src.posttraining.dataset import LAMBDA_CHAT_VALIDATION_SPLIT
from src.shared.model.transformer import DecoderOnlyTransformer


def save_chat_model(
    model: DecoderOnlyTransformer,
    model_dir: Path,
    model_config: dict[str, int | float],
    args: argparse.Namespace,
    pad_token_id: int,
    bos_token_id: int,
    eos_token_id: int,
    end_of_turn_token_id: int,
) -> None:
    # ---------------------------------------------------------
    # Save the final chat-tuned weights and metadata needed by
    # inference to rebuild the same architecture.
    # ---------------------------------------------------------
    torch.save(model.state_dict(), model_dir / "model.pth")

    # ---------------------------------------------------------
    # Persist posttraining provenance alongside architecture fields
    # inherited from the base model configuration.
    # ---------------------------------------------------------
    payload = {
        **model_config,
        "training_max_len": args.max_len,
        "learning_rate": args.learning_rate,
        "pad_token_id": pad_token_id,
        "bos_token_id": bos_token_id,
        "eos_token_id": eos_token_id,
        "end_of_turn_token_id": end_of_turn_token_id,
        "base_model_id": args.base_model_id,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "devices": getattr(args, "devices", "auto"),
        "device_count": getattr(args, "device_count", 1),
        "global_batch_size": getattr(args, "global_batch_size", getattr(args, "batch_size", 1)),
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "global_effective_batch_size": getattr(
            args,
            "global_effective_batch_size",
            getattr(args, "batch_size", 1),
        ),
        "lr_schedule": "warmup_cosine",
        "lr_warmup_steps": args.lr_warmup_steps,
        "min_learning_rate": args.min_learning_rate,
        "min_learning_rate_ratio": args.min_learning_rate_ratio,
        "loss_chunk_size": args.loss_chunk_size,
        "trainable_layers": "all",
        "chat_template_version": 1,
        "posttraining_datasets": [
            f"{LAMBDA_CHAT_DATASET_PATH}:{LAMBDA_CHAT_TRAIN_SPLIT}",
        ],
        "validation_dataset": f"{LAMBDA_CHAT_DATASET_PATH}:{LAMBDA_CHAT_VALIDATION_SPLIT}",
        "validation_cache_path": args.validation_cache_path,
        "validation_sample_count": args.validation_sample_count,
        "posttraining_steps": args.posttraining_steps,
    }

    with open(model_dir / "model_config.json", "w") as f:
        json.dump(payload, f)
