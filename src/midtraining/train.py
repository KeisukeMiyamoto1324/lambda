from hashlib import blake2b
import json
import os
from pathlib import Path
import sys

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.midtraining.cli import parse_args
from src.midtraining.training_corpus_cases import MIDTRAINING_TRAIN_CORPUS_CASE
from src.midtraining.training_corpus_cases import MIDTRAINING_VALIDATION_CORPUS_CASE
from src.midtraining.training_corpus_cases import serialize_midtraining_corpus_case
from src.shared.device_utils import is_global_zero_process
from src.shared.device_utils import resolve_device_count
from src.shared.device_utils import resolve_devices
from src.shared.device_utils import resolve_strategy
from src.shared.device_utils import wait_for_file
from src.shared.model.compilation import compile_training_model
from src.shared.model.float8_training import convert_model_to_float8_training
from src.shared.model.float8_training import FLOAT8_RECIPE
from src.shared.model.float8_training import TRAINING_PRECISION
from src.shared.packed_dataset import build_tokenized_cache
from src.shared.packed_dataset import LocalTokenizedDataset
from src.shared.packed_dataset import PackedCorpusDataset
from src.shared.pytorch_artifacts import load_model_config
from src.shared.pytorch_artifacts import load_pytorch_model
from src.shared.pytorch_artifacts import push_pytorch_model_artifacts
from src.shared.tokenizer import ByteLevelBPE
from src.shared.training_corpus import TrainingCorpusCase
from src.shared.training_checkpoint import resolve_resume_shuffle_seed
from src.shared.training_token_budget import show_training_token_budget
from src.shared.training_progress import FullTrainingProgressBar
from src.shared.validation_generation import ValidationGenerationCallback

from dotenv import load_dotenv


load_dotenv()


PACKING_VERSION = "bucket-packing-v1"
SHUFFLE_BUFFER_SIZE = 10000
SHUFFLE_SEED = 17


def build_dataset_signature(corpus_case: TrainingCorpusCase) -> str:
    # ---------------------------------------------------------
    # Hash one dataset configuration into a stable identifier for
    # model metadata and validation cache invalidation.
    # ---------------------------------------------------------
    payload = serialize_midtraining_corpus_case(corpus_case)
    encoded_payload = json.dumps(payload, sort_keys=True).encode("utf-8")
    return blake2b(encoded_payload, digest_size=8).hexdigest()


def main() -> None:
    # ---------------------------------------------------------
    # Load the pretrained artifacts and resolve the requested
    # context length before creating datasets or output files.
    # ---------------------------------------------------------
    args = parse_args()
    source_model_dir = Path(args.model_path)
    source_model_config = load_model_config(model_dir=source_model_dir)
    max_len = int(source_model_config["max_len"] if args.max_len is None else args.max_len)
    tokenizer = ByteLevelBPE.load(source_model_dir)
    devices = resolve_devices(devices=args.devices)
    device_count = resolve_device_count(devices=devices)
    strategy = resolve_strategy(device_count=device_count)
    pad_token_id = tokenizer.token_to_id(tokenizer.pad_token)
    bos_token_id = tokenizer.token_to_id(tokenizer.bos_token)
    eos_token_id = tokenizer.token_to_id(tokenizer.eos_token)
    min_learning_rate = args.learning_rate * args.min_learning_rate_ratio

    # ---------------------------------------------------------
    # Prepare output, validation cache, and deterministic corpus
    # partitions shared by the full step-bounded training run.
    # ---------------------------------------------------------
    model_dir = Path(args.output_path)
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = model_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    validation_sample_count = args.batch_size * args.val_batches * device_count
    train_dataset_signature = build_dataset_signature(
        corpus_case=MIDTRAINING_TRAIN_CORPUS_CASE,
    )
    validation_dataset_signature = build_dataset_signature(
        corpus_case=MIDTRAINING_VALIDATION_CORPUS_CASE,
    )
    default_validation_cache_path = (
        model_dir
        / f"validation-cache-{validation_dataset_signature}-{PACKING_VERSION}-len{max_len}"
        f"-samples{validation_sample_count}.pt"
    )
    validation_cache_path = (
        Path(args.validation_cache_path) if args.validation_cache_path else default_validation_cache_path
    )
    shuffle_seed = resolve_resume_shuffle_seed(
        base_seed=SHUFFLE_SEED,
        checkpoint_path=args.resume_from_checkpoint,
    )

    # ---------------------------------------------------------
    # Stream repeated corpus passes inside persistent workers until
    # Lightning reaches the configured optimizer step budget.
    # ---------------------------------------------------------
    train_dataset = PackedCorpusDataset(
        corpus_case=MIDTRAINING_TRAIN_CORPUS_CASE,
        tokenizer=tokenizer,
        max_len=max_len,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        shuffle_buffer_size=SHUFFLE_BUFFER_SIZE,
        shuffle_seed=shuffle_seed,
        repeat_forever=True,
    )
    validation_source_dataset = PackedCorpusDataset(
        corpus_case=MIDTRAINING_VALIDATION_CORPUS_CASE,
        tokenizer=tokenizer,
        max_len=max_len,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
    )
    validation_cache_metadata = {
        "packing_version": PACKING_VERSION,
        "validation_dataset_signature": validation_dataset_signature,
        "validation_dataset_case": serialize_midtraining_corpus_case(
            MIDTRAINING_VALIDATION_CORPUS_CASE
        ),
    }

    if not validation_cache_path.exists() and is_global_zero_process():
        build_tokenized_cache(
            dataset=validation_source_dataset,
            path=validation_cache_path,
            num_samples=validation_sample_count,
            max_len=max_len,
            metadata=validation_cache_metadata,
        )

    if not is_global_zero_process():
        wait_for_file(path=validation_cache_path)

    val_dataset = LocalTokenizedDataset(
        path=validation_cache_path,
        max_len=max_len,
        num_samples=validation_sample_count,
        metadata=validation_cache_metadata,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    # ---------------------------------------------------------
    # Rebuild the pretrained architecture and apply the same warmup
    # and cosine decay schedule used by pretraining.
    # ---------------------------------------------------------
    model, _ = load_pytorch_model(
        model_dir=source_model_dir,
        vocab_size=tokenizer.get_vocab_size(),
        learning_rate=args.learning_rate,
        max_len=max_len,
        lr_warmup_steps=args.lr_warmup_steps,
        lr_total_steps=args.max_steps,
        min_learning_rate=min_learning_rate,
    )

    # ---------------------------------------------------------
    # Convert the large decoder projections to tensorwise FP8 before
    # compiling the artifact-compatible Transformer body.
    # ---------------------------------------------------------
    model = convert_model_to_float8_training(model=model)
    model = compile_training_model(model=model)

    # ---------------------------------------------------------
    # Print the planned context-token budget and its ratio to the
    # model size once before the training loop starts.
    # ---------------------------------------------------------
    if is_global_zero_process():
        show_training_token_budget(
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            device_count=device_count,
            max_len=max_len,
            parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        )

    # ---------------------------------------------------------
    # Save resumable checkpoints at fixed step intervals, the best
    # validation checkpoint, and batched CSV metrics.
    # ---------------------------------------------------------
    callbacks = [
        FullTrainingProgressBar(),
        ValidationGenerationCallback(
            dataset=val_dataset,
            tokenizer=tokenizer,
            output_dir=model_dir / "validation-generations",
        ),
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="step-{step}",
            every_n_train_steps=args.checkpoint_every_n_steps,
            save_top_k=-1,
            save_on_train_epoch_end=False,
        ),
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="best-step={step}-val_loss={val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    metrics_logger = CSVLogger(
        save_dir=model_dir,
        name="metrics",
        version="",
        flush_logs_every_n_steps=args.metric_log_every_n_steps,
    )
    strategy_kwargs = {"strategy": strategy} if strategy is not None else {}
    trainer = L.Trainer(
        max_steps=args.max_steps,
        accelerator="cuda",
        devices=devices,
        precision="bf16-mixed",
        callbacks=callbacks,
        logger=metrics_logger,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        log_every_n_steps=args.metric_log_every_n_steps,
        val_check_interval=args.val_check_interval,
        limit_val_batches=args.val_batches,
        num_sanity_val_steps=0,
        enable_progress_bar=False,
        **strategy_kwargs,
    )
    checkpoint_path = args.resume_from_checkpoint or None
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=checkpoint_path,
    )

    # ---------------------------------------------------------
    # Save final weights with inherited architecture metadata and
    # the exact scheduled-LR, corpus, and step budget details.
    # ---------------------------------------------------------
    if not trainer.is_global_zero:
        return

    torch.save(model.state_dict(), model_dir / "model.pth")
    model_config = {
        **source_model_config,
        "max_len": max_len,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "devices": args.devices,
        "device_count": device_count,
        "global_batch_size": args.batch_size * device_count,
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "global_effective_batch_size": args.batch_size * args.gradient_accumulation_steps * device_count,
        "lr_schedule": "warmup_cosine",
        "lr_warmup_steps": args.lr_warmup_steps,
        "min_learning_rate": min_learning_rate,
        "min_learning_rate_ratio": args.min_learning_rate_ratio,
        "pad_token_id": pad_token_id,
        "bos_token_id": bos_token_id,
        "eos_token_id": eos_token_id,
        "midtraining_source_model": str(source_model_dir),
        "midtraining_max_steps": args.max_steps,
        "train_dataset_signature": train_dataset_signature,
        "validation_dataset_signature": validation_dataset_signature,
        "train_dataset_case": serialize_midtraining_corpus_case(
            MIDTRAINING_TRAIN_CORPUS_CASE
        ),
        "validation_dataset_case": serialize_midtraining_corpus_case(
            MIDTRAINING_VALIDATION_CORPUS_CASE
        ),
        "validation_cache_path": str(validation_cache_path),
        "validation_sample_count": validation_sample_count,
        "packing_version": PACKING_VERSION,
        "shuffle_buffer_size": SHUFFLE_BUFFER_SIZE,
        "base_shuffle_seed": SHUFFLE_SEED,
        "shuffle_seed": shuffle_seed,
        "trained_steps": trainer.global_step,
        "training_precision": TRAINING_PRECISION,
        "float8_recipe": FLOAT8_RECIPE,
    }

    # ---------------------------------------------------------
    # Remove obsolete fields from older mid-training artifacts while
    # keeping the active scheduler metadata written above.
    # ---------------------------------------------------------
    obsolete_config_keys = [
        "midtraining_epochs",
        "midtraining_completed_epochs",
    ]

    for key in obsolete_config_keys:
        model_config.pop(key, None)

    with open(model_dir / "model_config.json", "w") as file:
        json.dump(model_config, file)

    tokenizer.save_pretrained(path=model_dir)

    # ---------------------------------------------------------
    # Optionally publish the final PyTorch artifacts without
    # checkpoints, validation tensors, metrics, or source code.
    # ---------------------------------------------------------
    if args.push_to_hub:
        push_pytorch_model_artifacts(
            output_path=model_dir,
            repo_id=os.environ["HF_REPO"],
            private=True,
            commit_message="Upload mid-trained MyLLM model",
        )


if __name__ == "__main__":
    main()
