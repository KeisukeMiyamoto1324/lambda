import argparse
import json
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from rich.table import Table

from src.eval.JMMLU.dataset import JMMLU_DATASET_ID
from src.eval.JMMLU.dataset import JmmluExample
from src.eval.JMMLU.dataset import download_jmmlu_archive
from src.eval.JMMLU.dataset import load_examples
from src.eval.JMMLU.models import ChoiceScorer
from src.eval.JMMLU.models import load_choice_scorer
from src.eval.JMMLU.scoring import predict_answer
from src.shared.console import console
from src.shared.console import progress_manager


DEFAULT_OUTPUT_DIR = Path("eval_results/jmmlu")


@dataclass(frozen=True)
class SubjectResult:
    subject: str
    accuracy: float
    correct: int
    total: int


@dataclass(frozen=True)
class EvaluationResult:
    model_source: str
    backend: str
    dataset: str
    scoring_method: str
    device: str
    torch_dtype: str
    overall: SubjectResult
    by_subject: list[SubjectResult]


def run_evaluation(args: argparse.Namespace) -> None:
    # ---------------------------------------------------------
    # Load the selected model scorer, JMMLU examples, run
    # evaluation, then print and save the final metrics.
    # ---------------------------------------------------------
    scorer = load_choice_scorer(
        model_source=args.model,
        backend=args.backend,
        torch_dtype_name=args.torch_dtype,
        trust_remote_code=args.trust_remote_code,
    )

    archive_path = download_jmmlu_archive()
    subjects = None if args.subjects is None else [str(subject) for subject in args.subjects]
    examples = load_examples(archive_path=archive_path, subjects=subjects)
    selected_examples = examples if args.limit is None else examples[: args.limit]

    if not selected_examples:
        raise ValueError("No JMMLU examples were selected")

    result = evaluate_examples(
        scorer=scorer,
        examples=selected_examples,
    )
    render_result(result=result)
    save_result(result=result, output_path=resolve_output_json(output_json=args.output_json))


def evaluate_examples(
    scorer: ChoiceScorer,
    examples: list[JmmluExample],
) -> EvaluationResult:
    # ---------------------------------------------------------
    # Evaluate all selected examples while tracking both overall
    # and per-subject accuracy counts.
    # ---------------------------------------------------------
    subject_counts: dict[str, dict[str, int]] = {}
    task_id = progress_manager.add_task(description="JMMLU", total=len(examples))

    try:
        for index, example in enumerate(examples, start=1):
            prediction = predict_answer(
                scorer=scorer,
                example=example,
            )
            subject_count = subject_counts.setdefault(example.subject, {"correct": 0, "total": 0})
            subject_count["correct"] += int(prediction == example.answer)
            subject_count["total"] += 1

            correct = sum(counts["correct"] for counts in subject_counts.values())
            progress_manager.update(
                task_id=task_id,
                advance=1,
                metrics=f"accuracy={correct / index:.4f}",
            )
    finally:
        progress_manager.finish_task(task_id=task_id)

    return build_evaluation_result(
        scorer=scorer,
        subject_counts=subject_counts,
    )


def build_evaluation_result(
    scorer: ChoiceScorer,
    subject_counts: dict[str, dict[str, int]],
) -> EvaluationResult:
    # ---------------------------------------------------------
    # Convert raw counters into serializable result dataclasses
    # for terminal rendering and JSON output.
    # ---------------------------------------------------------
    by_subject = [
        SubjectResult(
            subject=subject,
            accuracy=counts["correct"] / counts["total"],
            correct=counts["correct"],
            total=counts["total"],
        )
        for subject, counts in sorted(subject_counts.items())
    ]
    total = sum(result.total for result in by_subject)
    correct = sum(result.correct for result in by_subject)
    overall = SubjectResult(
        subject="overall",
        accuracy=correct / total,
        correct=correct,
        total=total,
    )
    return EvaluationResult(
        model_source=scorer.model_source,
        backend=scorer.backend,
        dataset=JMMLU_DATASET_ID,
        scoring_method="zero_shot_mmlu_answer_label_log_likelihood",
        device=scorer.device_name,
        torch_dtype=scorer.torch_dtype_name,
        overall=overall,
        by_subject=by_subject,
    )


def render_result(result: EvaluationResult) -> None:
    # ---------------------------------------------------------
    # Print overall and subject-level metrics with Rich tables so
    # terminal output stays easy to scan.
    # ---------------------------------------------------------
    console.print("[bold cyan]JMMLU result[/bold cyan]")
    console.print(f"model: {result.model_source}")
    console.print(f"backend: {result.backend}")
    console.print(f"accuracy: {result.overall.accuracy:.4f}")
    console.print(f"correct: {result.overall.correct}")
    console.print(f"total: {result.overall.total}")

    table = Table(title="JMMLU by subject")
    table.add_column("subject")
    table.add_column("accuracy", justify="right")
    table.add_column("correct", justify="right")
    table.add_column("total", justify="right")

    for subject_result in result.by_subject:
        table.add_row(
            subject_result.subject,
            f"{subject_result.accuracy:.4f}",
            str(subject_result.correct),
            str(subject_result.total),
        )

    console.print(table)


def resolve_output_json(output_json: str | None) -> Path:
    # ---------------------------------------------------------
    # Use an explicit output path when provided. Otherwise, create
    # a timestamped result file under eval_results.
    # ---------------------------------------------------------
    if output_json is not None:
        return Path(output_json)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return DEFAULT_OUTPUT_DIR / f"{timestamp}.json"


def save_result(result: EvaluationResult, output_path: Path) -> None:
    # ---------------------------------------------------------
    # Persist the evaluation summary as UTF-8 JSON for experiment
    # tracking outside the terminal output.
    # ---------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(asdict(result), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    console.print(f"[cyan]saved json[/cyan] {output_path}")
