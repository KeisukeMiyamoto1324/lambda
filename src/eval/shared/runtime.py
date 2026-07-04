import json
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from typing import Protocol
from typing import TypeVar

from src.eval.shared.models import ChoiceScorer
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


def select_examples(examples: list[ExampleT], limit: int | None, benchmark_name: str) -> list[ExampleT]:
    # ---------------------------------------------------------
    # Apply an optional limit and reject empty benchmark runs
    # before model scoring starts.
    # ---------------------------------------------------------
    selected_examples = examples if limit is None else examples[:limit]

    if not selected_examples:
        raise ValueError(f"No {benchmark_name} examples were selected")

    return selected_examples


def count_correct_predictions(
    scorer: ChoiceScorer,
    examples: list[ExampleT],
    benchmark_name: str,
    predict_answer: Callable[[ChoiceScorer, ExampleT], str],
) -> AccuracyResult:
    # ---------------------------------------------------------
    # Run exact-match multiple-choice evaluation with shared
    # progress reporting and overall accuracy counts.
    # ---------------------------------------------------------
    correct = 0
    task_id = progress_manager.add_task(description=benchmark_name, total=len(examples))

    try:
        for index, example in enumerate(examples, start=1):
            prediction = predict_answer(scorer, example)
            correct += int(prediction == example.answer)
            progress_manager.update(
                task_id=task_id,
                advance=1,
                metrics=f"accuracy={correct / index:.4f}",
            )
    finally:
        progress_manager.finish_task(task_id=task_id)

    return AccuracyResult(
        accuracy=correct / len(examples),
        correct=correct,
        total=len(examples),
    )


def save_json_result(result: object, output_path: Path) -> None:
    # ---------------------------------------------------------
    # Persist a dataclass result summary as UTF-8 JSON for
    # experiment tracking outside the terminal output.
    # ---------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(asdict(result), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    console.print(f"[cyan]saved json[/cyan] {output_path}")
