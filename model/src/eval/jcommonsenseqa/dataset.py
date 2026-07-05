from dataclasses import dataclass
from typing import Any

from datasets import load_dataset


JCOMMONSENSEQA_DATASET_ID = "sbintuitions/JCommonsenseQA"
JCOMMONSENSEQA_CONFIG = "default"
ANSWER_LABELS = ("0", "1", "2", "3", "4")


@dataclass(frozen=True)
class JCommonsenseQAExample:
    q_id: int
    question: str
    choices: list[str]
    answer: str


def load_examples(split: str) -> list[JCommonsenseQAExample]:
    # ---------------------------------------------------------
    # Load the public Hugging Face dataset split and convert rows
    # into a stable internal multiple-choice shape.
    # ---------------------------------------------------------
    dataset = load_dataset(JCOMMONSENSEQA_DATASET_ID, JCOMMONSENSEQA_CONFIG, split=split)
    rows = [dict(row) for row in dataset]
    return [build_example(row=row) for row in rows]


def build_example(row: dict[str, Any]) -> JCommonsenseQAExample:
    # ---------------------------------------------------------
    # Convert one JCommonsenseQA row. The official choices are
    # choice0 through choice4 and label is an integer index.
    # ---------------------------------------------------------
    return JCommonsenseQAExample(
        q_id=int(row["q_id"]),
        question=str(row["question"]),
        choices=[str(row[f"choice{index}"]) for index in range(len(ANSWER_LABELS))],
        answer=str(row["label"]),
    )
