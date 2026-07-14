import math

from torch.utils.data import DataLoader

from src.posttraining.dataset import LAMBDA_CHAT_TRAIN_SPLIT
from src.posttraining.dataset import LAMBDA_CHAT_VALIDATION_SPLIT
from src.posttraining.dataset import LambdaChatDataset
from src.shared.tokenizer import ByteLevelBPE


def build_dataloaders(
    tokenizer: ByteLevelBPE,
    max_len: int,
    batch_size: int,
    num_workers: int,
    accelerator: str,
    repeat_epochs: int,
    gradient_accumulation_steps: int = 1,
    device_count: int = 1,
) -> tuple[DataLoader, DataLoader, int]:
    # ---------------------------------------------------------
    # Resolve tokenizer ids shared by both SFT datasets and the
    # Transformer loss masking convention.
    # ---------------------------------------------------------
    pad_token_id = tokenizer.token_to_id(tokenizer.pad_token)
    bos_token_id = tokenizer.token_to_id(tokenizer.bos_token)
    eos_token_id = tokenizer.token_to_id(tokenizer.eos_token)
    end_of_turn_token_id = tokenizer.token_to_id(tokenizer.end_of_turn_token)

    # ---------------------------------------------------------
    # Build fixed lambda-chat train and validation datasets from the
    # official train/validation splits.
    # ---------------------------------------------------------
    train_dataset = LambdaChatDataset(
        tokenizer=tokenizer,
        split=LAMBDA_CHAT_TRAIN_SPLIT,
        max_len=max_len,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        end_of_turn_token_id=end_of_turn_token_id,
    )
    validation_dataset = LambdaChatDataset(
        tokenizer=tokenizer,
        split=LAMBDA_CHAT_VALIDATION_SPLIT,
        max_len=max_len,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        end_of_turn_token_id=end_of_turn_token_id,
    )
    use_pin_memory = accelerator == "cuda"
    use_persistent_workers = num_workers > 0

    # ---------------------------------------------------------
    # Wrap datasets with DataLoaders configured consistently with
    # the existing pretraining pipeline.
    # ---------------------------------------------------------
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
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
    samples_per_step = batch_size * gradient_accumulation_steps * device_count
    max_steps = math.ceil(len(train_dataset) / samples_per_step) * repeat_epochs
    return train_dataloader, validation_dataloader, max_steps
