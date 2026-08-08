from src.shared.training_corpus import serialize_training_corpus_case
from src.shared.training_corpus import TrainingCorpusCase


PretrainingCorpusCase = TrainingCorpusCase


PRETRAINING_TRAIN_CORPUS_CASE = PretrainingCorpusCase(
    name="lambda-corpus-train",
    genre="mixed",
    language="ja",
    dataset_path="KeisukeMiyamoto/lambda-corpus",
    config_name="default",
    split="train",
    text_column="text",
)


PRETRAINING_VALIDATION_CORPUS_CASE = PretrainingCorpusCase(
    name="lambda-corpus-validation",
    genre="mixed",
    language="ja",
    dataset_path="KeisukeMiyamoto/lambda-corpus",
    config_name="default",
    split="validation",
    text_column="text",
)


def serialize_pretraining_corpus_case(
    corpus_case: PretrainingCorpusCase,
) -> dict[str, str]:
    # ---------------------------------------------------------
    # Convert the corpus case to a JSON-compatible dictionary so
    # training artifacts can record the exact dataset source.
    # ---------------------------------------------------------
    return serialize_training_corpus_case(corpus_case=corpus_case)
