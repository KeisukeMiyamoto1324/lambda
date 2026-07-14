from datasets import load_dataset
from torch.utils.data import Dataset

from src.posttraining.chat_template import ChatMessage
from src.posttraining.chat_template import tokenize_chat_messages
from src.shared.packed_dataset import BucketSequencePacker
from src.shared.packed_dataset import PackedTrainingExample
from src.shared.tokenizer import ByteLevelBPE


LAMBDA_CHAT_DATASET_PATH = "KeisukeMiyamoto/lambda-chat"
LAMBDA_CHAT_TRAIN_SPLIT = "train"
LAMBDA_CHAT_VALIDATION_SPLIT = "validation"


class LambdaChatDataset(Dataset[PackedTrainingExample]):
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
        # Load lambda-chat records locally and prepare the shared
        # deterministic packer for the selected dataset split.
        # ---------------------------------------------------------
        dataset = load_dataset(path=LAMBDA_CHAT_DATASET_PATH, split=split)
        packer = BucketSequencePacker(
            max_len=max_len,
            pad_token_id=pad_token_id,
            source_name=f"{LAMBDA_CHAT_DATASET_PATH}:{split}",
        )
        self.examples: list[PackedTrainingExample] = []

        # ---------------------------------------------------------
        # Tokenize every conversation without padding and add it to
        # the bounded best-fit window used by pretraining.
        # ---------------------------------------------------------
        for sample in dataset:
            input_token_ids, label_token_ids = build_chat_segment(
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
            packed_example = packer.add_segment(
                input_token_ids=input_token_ids,
                label_token_ids=label_token_ids,
            )

            if packed_example is not None:
                self.examples.append(packed_example)

        # ---------------------------------------------------------
        # Materialize the remaining packed sequences so DataLoader
        # length and repeat-epoch step counts are exact.
        # ---------------------------------------------------------
        self.examples.extend(packer.drain())

    def __len__(self) -> int:
        # ---------------------------------------------------------
        # Return the number of packed lambda-chat sequences available
        # from the selected split.
        # ---------------------------------------------------------
        return len(self.examples)

    def __getitem__(self, index: int) -> PackedTrainingExample:
        # ---------------------------------------------------------
        # Return one fixed-length packed chat example without any
        # additional network, tokenizer, or packing work.
        # ---------------------------------------------------------
        return self.examples[index]


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
