import torch
from datasets import load_dataset
from torch.utils.data import Dataset

from src.posttraining.chat_template import ChatMessage
from src.posttraining.chat_template import tokenize_chat_messages
from src.shared.tokenizer import ByteLevelBPE


LAMBDA_CHAT_DATASET_PATH = "KeisukeMiyamoto/lambda-chat"
LAMBDA_CHAT_TRAIN_SPLIT = "train"
LAMBDA_CHAT_VALIDATION_SPLIT = "validation"


class LambdaChatDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        tokenizer: ByteLevelBPE,
        split: str,
        max_len: int,
        pad_token_id: int,
        bos_token_id: int,
        eos_token_id: int,
        end_of_turn_token_id: int,
    ) -> None:
        super().__init__()

        # ---------------------------------------------------------
        # Load lambda-chat records locally so the train split can be
        # reused for several epochs.
        # ---------------------------------------------------------
        dataset = load_dataset(path=LAMBDA_CHAT_DATASET_PATH, split=split)
        self.examples = [
            build_tensor_example(
                tokenizer=tokenizer,
                messages=[
                    ChatMessage(role=message["role"], content=message["content"])
                    for message in sample["messages"]
                ],
                max_len=max_len,
                pad_token_id=pad_token_id,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                end_of_turn_token_id=end_of_turn_token_id,
            )
            for sample in dataset
        ]

    def __len__(self) -> int:
        # ---------------------------------------------------------
        # Return the number of loaded lambda-chat examples
        # available in the selected split.
        # ---------------------------------------------------------
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # ---------------------------------------------------------
        # Return one pre-tokenized fixed-length chat example without
        # additional network or tokenizer work.
        # ---------------------------------------------------------
        return self.examples[index]


def build_tensor_example(
    tokenizer: ByteLevelBPE,
    messages: list[ChatMessage],
    max_len: int,
    pad_token_id: int,
    bos_token_id: int,
    eos_token_id: int,
    end_of_turn_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # ---------------------------------------------------------
    # Tokenize one chat record through the shared template and
    # convert its two streams into tensors for model training.
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
    input_ids = torch.tensor(example.input_ids, dtype=torch.long)
    labels = torch.tensor(example.labels, dtype=torch.long)
    return input_ids, labels
