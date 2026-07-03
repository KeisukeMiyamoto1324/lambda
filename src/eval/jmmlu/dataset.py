from dataclasses import dataclass
from pathlib import Path
import zipfile

import pandas as pd
from huggingface_hub import hf_hub_download


JMMLU_DATASET_ID = "nlp-waseda/JMMLU"
JMMLU_ARCHIVE_NAME = "JMMLU.zip"
ANSWER_LABELS = ("A", "B", "C", "D")


@dataclass(frozen=True)
class JmmluExample:
    subject: str
    question: str
    choices: list[str]
    answer: str


def download_jmmlu_archive() -> Path:
    # ---------------------------------------------------------
    # Download the JMMLU archive directly because the current
    # datasets package no longer runs this dataset script.
    # ---------------------------------------------------------
    return Path(
        hf_hub_download(
            repo_id=JMMLU_DATASET_ID,
            filename=JMMLU_ARCHIVE_NAME,
            repo_type="dataset",
        )
    )


def load_examples(archive_path: Path, subjects: list[str] | None) -> list[JmmluExample]:
    # ---------------------------------------------------------
    # Read subject CSV files from the official JMMLU zip and
    # convert each row into a stable internal example shape.
    # ---------------------------------------------------------
    with zipfile.ZipFile(archive_path) as archive:
        csv_names = [
            name
            for name in archive.namelist()
            if name.startswith("JMMLU/test/") and name.endswith(".csv")
        ]
        selected_csv_names = [
            name
            for name in csv_names
            if subjects is None or Path(name).stem in subjects
        ]
        examples = [
            example
            for csv_name in selected_csv_names
            for example in load_subject_examples(archive=archive, csv_name=csv_name)
        ]

    if subjects is not None:
        validate_subjects(subjects=subjects, examples=examples)

    return examples


def load_subject_examples(archive: zipfile.ZipFile, csv_name: str) -> list[JmmluExample]:
    # ---------------------------------------------------------
    # Parse one JMMLU subject CSV. The official columns are
    # question, A, B, C, D, and answer.
    # ---------------------------------------------------------
    subject = Path(csv_name).stem

    with archive.open(csv_name) as csv_file:
        data_frame = pd.read_csv(csv_file, encoding="utf-8-sig")

    return [
        JmmluExample(
            subject=subject,
            question=str(row["question"]),
            choices=[str(row[label]) for label in ANSWER_LABELS],
            answer=str(row["answer"]),
        )
        for row in data_frame.to_dict(orient="records")
    ]


def validate_subjects(subjects: list[str], examples: list[JmmluExample]) -> None:
    # ---------------------------------------------------------
    # Reject subject names that were not found in the official
    # JMMLU archive so typos do not silently change the result.
    # ---------------------------------------------------------
    found_subjects = {example.subject for example in examples}
    missing_subjects = sorted(set(subjects) - found_subjects)

    if missing_subjects:
        raise ValueError(f"Unknown JMMLU subjects: {', '.join(missing_subjects)}")
