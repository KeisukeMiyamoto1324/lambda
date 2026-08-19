from src.shared.training_corpus import serialize_training_corpus_case
from src.shared.training_corpus import TrainingCorpusCase


MIDTRAINING_TRAIN_CORPUS_CASE = TrainingCorpusCase(
    name="synthetic-textbook-jp-train",
    genre="textbook",
    language="ja",
    dataset_path="KeisukeMiyamoto/SyntheticTextbook-jp",
    config_name="default",
    split="train",
    text_column="rewrite",
)


MIDTRAINING_VALIDATION_CORPUS_CASE = TrainingCorpusCase(
    name="synthetic-textbook-jp-validation",
    genre="textbook",
    language="ja",
    dataset_path="KeisukeMiyamoto/SyntheticTextbook-jp",
    config_name="default",
    split="validation",
    text_column="rewrite",
)


def serialize_midtraining_corpus_case(
    corpus_case: TrainingCorpusCase,
) -> dict[str, str]:
    # ---------------------------------------------------------
    # Serialize one mid-training corpus definition for validation
    # cache invalidation and model artifact metadata.
    # ---------------------------------------------------------
    return serialize_training_corpus_case(corpus_case=corpus_case)
