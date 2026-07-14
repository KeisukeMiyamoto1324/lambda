from pathlib import Path

from torch.utils.data import DataLoader

from src.posttraining.dataset import LAMBDA_CHAT_DATASET_PATH
from src.posttraining.dataset import LAMBDA_CHAT_TRAIN_SPLIT
from src.posttraining.dataset import LAMBDA_CHAT_VALIDATION_SPLIT
from src.posttraining.dataset import LambdaChatDataset
from src.shared.device_utils import is_global_zero_process
from src.shared.device_utils import wait_for_file
from src.shared.packed_dataset import build_tokenized_cache
from src.shared.packed_dataset import LocalTokenizedDataset
from src.shared.tokenizer import ByteLevelBPE


PACKING_VERSION = "bucket-packing-v1"
SHUFFLE_BUFFER_SIZE = 10000
SHUFFLE_SEED = 17


def build_dataloaders(
    tokenizer: ByteLevelBPE,
    max_len: int,
    batch_size: int,
    num_workers: int,
    accelerator: str,
    validation_cache_path: Path,
    validation_sample_count: int,
    shuffle_seed: int,
) -> tuple[DataLoader, DataLoader]:
    # ---------------------------------------------------------
    # Resolve tokenizer ids shared by streamed training records and
    # the fixed local validation cache.
    # ---------------------------------------------------------
    pad_token_id = tokenizer.token_to_id(tokenizer.pad_token)
    bos_token_id = tokenizer.token_to_id(tokenizer.bos_token)
    eos_token_id = tokenizer.token_to_id(tokenizer.eos_token)
    end_of_turn_token_id = tokenizer.token_to_id(tokenizer.end_of_turn_token)
    dataset_kwargs = {
        "tokenizer": tokenizer,
        "max_len": max_len,
        "pad_token_id": pad_token_id,
        "bos_token_id": bos_token_id,
        "eos_token_id": eos_token_id,
        "end_of_turn_token_id": end_of_turn_token_id,
    }

    # ---------------------------------------------------------
    # Stream and repeat training data while keeping shuffle work in
    # DataLoader workers instead of blocking program startup.
    # ---------------------------------------------------------
    train_dataset = LambdaChatDataset(
        split=LAMBDA_CHAT_TRAIN_SPLIT,
        shuffle_buffer_size=SHUFFLE_BUFFER_SIZE,
        shuffle_seed=shuffle_seed,
        repeat_forever=True,
        **dataset_kwargs,
    )
    validation_source_dataset = LambdaChatDataset(
        split=LAMBDA_CHAT_VALIDATION_SPLIT,
        **dataset_kwargs,
    )
    validation_cache_metadata = {
        "packing_version": PACKING_VERSION,
        "dataset_path": LAMBDA_CHAT_DATASET_PATH,
        "dataset_split": LAMBDA_CHAT_VALIDATION_SPLIT,
    }

    # ---------------------------------------------------------
    # Build validation tensors once on rank zero so repeated checks
    # never download or tokenize validation conversations again.
    # ---------------------------------------------------------
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

    validation_dataset = LocalTokenizedDataset(
        path=validation_cache_path,
        max_len=max_len,
        num_samples=validation_sample_count,
        metadata=validation_cache_metadata,
    )
    use_pin_memory = accelerator == "cuda"
    use_persistent_workers = num_workers > 0

    # ---------------------------------------------------------
    # Let workers consume the train stream in parallel and serve
    # validation batches from the local tensor cache.
    # ---------------------------------------------------------
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=use_persistent_workers,
    )
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=use_persistent_workers,
    )
    return train_dataloader, validation_dataloader
