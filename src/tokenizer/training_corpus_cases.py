from dataclasses import dataclass


@dataclass
class TrainingCorpusCase:
    name: str
    genre: str
    language: str
    dataset_path: str
    config_name: str
    split: str
    text_column: str
    sample_count: int
    max_chars: int


TRAINING_CORPUS_CASES = [
    TrainingCorpusCase(
        name="lambda-corpus",
        genre="mixed",
        language="ja",
        dataset_path="KeisukeMiyamoto/lambda-corpus",
        config_name="default",
        split="train",
        text_column="text",
        sample_count=51200,
        max_chars=8192,
    ),
]
