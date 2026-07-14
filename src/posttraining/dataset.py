from collections.abc import Iterator

import torch
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from torch.utils.data import IterableDataset

from src.posttraining.chat_template import ChatMessage
from src.posttraining.chat_template import tokenize_chat_messages
from src.shared.packed_dataset import BucketSequencePacker
from src.shared.packed_dataset import PackedTrainingExample
from src.shared.tokenizer import ByteLevelBPE


LAMBDA_CHAT_DATASET_PATH = "KeisukeMiyamoto/lambda-chat"
LAMBDA_CHAT_TRAIN_SPLIT = "train"
LAMBDA_CHAT_VALIDATION_SPLIT = "validation"


class LambdaChatDataset(IterableDataset[PackedTrainingExample]):
    def __init__(
        self,
        tokenizer: ByteLevelBPE,
        split: str,
        max_len: int,
        pad_token_id: int,
        bos_token_id: int,
        eos_token_id: int,
        end_of_turn_token_id: int,
        shuffle_buffer_size: int = 0,
        shuffle_seed: int = 0,
        repeat_forever: bool = False,
    ) -> None:
        super().__init__()

        # ---------------------------------------------------------
        # Keep only the settings needed to stream, tokenize, and pack
        # lambda-chat records inside DataLoader workers.
        # ---------------------------------------------------------
        self.tokenizer = tokenizer
        self.split = split
        self.max_len = max_len
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.end_of_turn_token_id = end_of_turn_token_id
        self.shuffle_buffer_size = shuffle_buffer_size
        self.shuffle_seed = shuffle_seed
        self.repeat_forever = repeat_forever

    def __iter__(self) -> Iterator[PackedTrainingExample]:
        # ---------------------------------------------------------
        # Repeat independently shuffled streaming passes until the
        # trainer reaches its configured maximum step count.
        # ---------------------------------------------------------
        pass_index = 0

        while True:
            yield from self._iter_split_pass(pass_index=pass_index)
            pass_index += 1

            if not self.repeat_forever:
                break

    def _iter_split_pass(self, pass_index: int) -> Iterator[PackedTrainingExample]:
        # ---------------------------------------------------------
        # Open only the requested split as a stream so startup does
        # not tokenize or retain the full dataset in memory.
        # ---------------------------------------------------------
        dataset = load_dataset(
            path=LAMBDA_CHAT_DATASET_PATH,
            split=self.split,
            streaming=True,
        )
        dataset = dataset.select_columns(["messages"])
        dataset = dataset.reshard()

        if self.shuffle_buffer_size > 0:
            dataset = dataset.shuffle(
                seed=self.shuffle_seed + pass_index,
                buffer_size=self.shuffle_buffer_size,
            )

        # ---------------------------------------------------------
        # Keep CUDA ranks on separate stream partitions. Hugging Face
        # handles DataLoader worker partitioning inside each rank.
        # ---------------------------------------------------------
        rank_count, rank_index = self._resolve_rank_partition()

        if rank_count > 1:
            dataset = split_dataset_by_node(
                dataset,
                rank=rank_index,
                world_size=rank_count,
            )

        packer = BucketSequencePacker(
            max_len=self.max_len,
            pad_token_id=self.pad_token_id,
            source_name=f"{LAMBDA_CHAT_DATASET_PATH}:{self.split}",
        )

        # ---------------------------------------------------------
        # Tokenize conversations as they arrive and emit packed fixed
        # length examples from the bounded packing buffer.
        # ---------------------------------------------------------
        for sample in dataset:
            input_token_ids, label_token_ids = build_chat_segment(
                tokenizer=self.tokenizer,
                messages=[
                    ChatMessage(role=message["role"], content=message["content"])
                    for message in sample["messages"]
                ],
                max_len=self.max_len,
                pad_token_id=self.pad_token_id,
                bos_token_id=self.bos_token_id,
                eos_token_id=self.eos_token_id,
                end_of_turn_token_id=self.end_of_turn_token_id,
            )
            packed_example = packer.add_segment(
                input_token_ids=input_token_ids,
                label_token_ids=label_token_ids,
            )

            if packed_example is not None:
                yield packed_example

        yield from packer.drain()

    def _resolve_rank_partition(self) -> tuple[int, int]:
        # ---------------------------------------------------------
        # Read distributed state only when iteration begins because
        # Lightning initializes CUDA ranks after DataLoader creation.
        # ---------------------------------------------------------
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return 1, 0

        return torch.distributed.get_world_size(), torch.distributed.get_rank()


def build_chat_segment(
    tokenizer: ByteLevelBPE,
    messages: list[ChatMessage],
    max_len: int,
    pad_token_id: int,
    bos_token_id: int,
    eos_token_id: int,
    end_of_turn_token_id: int,
) -> tuple[list[int], list[int]]:
    # ---------------------------------------------------------
    # Tokenize one chat record through the shared template and
    # return unpadded streams for cross-conversation packing.
    # ---------------------------------------------------------
    example = tokenize_chat_messages(
        tokenizer=tokenizer,
        messages=messages,
        max_len=max_len,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        end_of_turn_token_id=end_of_turn_token_id,
    )
    return example.input_ids, example.labels
