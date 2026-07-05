import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from typing import Generic
from typing import Protocol
from typing import TypeVar

from src.eval.shared.scorer_types import ChoiceScorer
from src.eval.shared.multiple_choice import MultipleChoicePrediction
from src.shared.console import console
from src.shared.console import progress_manager


class AnsweredExample(Protocol):
    answer: str


@dataclass(frozen=True)
class AccuracyResult:
    accuracy: float
    correct: int
    total: int


ExampleT = TypeVar("ExampleT", bound=AnsweredExample)


@dataclass(frozen=True)
class ExamplePrediction(Generic[ExampleT]):
    index: int
    example: ExampleT
    prediction: str
    losses: list[float]
    correct: bool


def select_examples(examples: list[ExampleT], limit: int | None, benchmark_name: str) -> list[ExampleT]:
    # ---------------------------------------------------------
    # Apply an optional limit and reject empty benchmark runs
    # before model scoring starts.
    # ---------------------------------------------------------
    selected_examples = examples if limit is None else examples[:limit]

    if not selected_examples:
        raise ValueError(f"No {benchmark_name} examples were selected")

    return selected_examples


def collect_predictions(
    scorer: ChoiceScorer,
    examples: list[ExampleT],
    benchmark_name: str,
    predict_answer: Callable[[ChoiceScorer, ExampleT], MultipleChoicePrediction],
) -> tuple[AccuracyResult, list[ExamplePrediction[ExampleT]]]:
    # ---------------------------------------------------------
    # Run exact-match multiple-choice evaluation while keeping
    # per-example predictions and per-choice losses.
    # ---------------------------------------------------------
    correct = 0
    predictions: list[ExamplePrediction[ExampleT]] = []
    task_id = progress_manager.add_task(description=benchmark_name, total=len(examples))

    try:
        for index, example in enumerate(examples, start=1):
            prediction = predict_answer(scorer, example)
            is_correct = prediction.prediction == example.answer
            correct += int(is_correct)
            predictions.append(
                ExamplePrediction(
                    index=index,
                    example=example,
                    prediction=prediction.prediction,
                    losses=prediction.losses,
                    correct=is_correct,
                )
            )
            progress_manager.update(
                task_id=task_id,
                advance=1,
                metrics=f"accuracy={correct / index:.4f}",
            )
    finally:
        progress_manager.finish_task(task_id=task_id)

    return (
        AccuracyResult(
            accuracy=correct / len(examples),
            correct=correct,
            total=len(examples),
        ),
        predictions,
    )


def build_output_dir(base_dir: Path, output_dir: str | None, model_source: str, timestamp: str) -> Path:
    # ---------------------------------------------------------
    # Resolve the output directory from an explicit CLI value or
    # a stable benchmark/model/timestamp directory name.
    # ---------------------------------------------------------
    if output_dir is not None:
        return Path(output_dir)

    safe_model_name = "".join(character if character.isalnum() else "-" for character in model_source)
    safe_model_name = "-".join(part for part in safe_model_name.split("-") if part)
    return base_dir / f"{safe_model_name}-{timestamp}"


def save_json_file(payload: object, output_path: Path) -> None:
    # ---------------------------------------------------------
    # Persist JSON payloads as UTF-8 files for experiment
    # tracking outside the terminal output.
    # ---------------------------------------------------------
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def save_csv_file(rows: list[dict[str, object]], output_path: Path) -> None:
    # ---------------------------------------------------------
    # Persist all per-example evaluation rows with a stable header
    # order based on the first row.
    # ---------------------------------------------------------
    if not rows:
        raise ValueError("CSV rows must not be empty")

    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_evaluation_files(config: object, rows: list[dict[str, object]], output_dir: Path) -> None:
    # ---------------------------------------------------------
    # Save one evaluation run as config.json plus result.csv in
    # a single output directory.
    # ---------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json_file(payload=config, output_path=output_dir / "config.json")
    save_csv_file(rows=rows, output_path=output_dir / "result.csv")
    console.print(f"[cyan]saved evaluation[/cyan] {output_dir}")
