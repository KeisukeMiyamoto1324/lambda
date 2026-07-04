import argparse
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from rich.table import Table

from src.eval.jmmlu.dataset import JMMLU_DATASET_ID
from src.eval.jmmlu.dataset import JmmluExample
from src.eval.jmmlu.dataset import download_jmmlu_archive
from src.eval.jmmlu.dataset import load_examples
from src.eval.jmmlu.scoring import predict_answer
from src.eval.shared.models import ChoiceScorer
from src.eval.shared.models import load_choice_scorer
from src.eval.shared.runtime import AccuracyResult
from src.eval.shared.runtime import ExamplePrediction
from src.eval.shared.runtime import build_output_dir
from src.eval.shared.runtime import collect_predictions
from src.eval.shared.runtime import save_evaluation_files
from src.eval.shared.runtime import select_examples
from src.shared.console import console


DEFAULT_OUTPUT_DIR = Path("eval_results/jmmlu")


@dataclass(frozen=True)
class SubjectResult(AccuracyResult):
    subject: str


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
    rows: list[ExamplePrediction[JmmluExample]]


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
    selected_examples = select_examples(
        examples=examples,
        limit=args.limit,
        benchmark_name="JMMLU",
    )

    result = evaluate_examples(
        scorer=scorer,
        examples=selected_examples,
    )
    render_result(result=result)
    output_dir = resolve_output_dir(output_dir=args.output_dir, model_source=args.model)
    save_result(
        result=result,
        output_dir=output_dir,
        limit=args.limit,
        subjects=subjects,
    )


def evaluate_examples(
    scorer: ChoiceScorer,
    examples: list[JmmluExample],
) -> EvaluationResult:
    # ---------------------------------------------------------
    # Evaluate all selected examples while tracking both overall
    # and per-subject accuracy counts.
    # ---------------------------------------------------------
    _, rows = collect_predictions(
        scorer=scorer,
        examples=examples,
        benchmark_name="JMMLU",
        predict_answer=predict_answer,
    )

    return build_evaluation_result(
        scorer=scorer,
        rows=rows,
    )


def build_evaluation_result(
    scorer: ChoiceScorer,
    rows: list[ExamplePrediction[JmmluExample]],
) -> EvaluationResult:
    # ---------------------------------------------------------
    # Convert raw counters into serializable result dataclasses
    # for terminal rendering and JSON output.
    # ---------------------------------------------------------
    subject_counts = build_subject_counts(rows=rows)
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
        rows=rows,
    )


def build_subject_counts(rows: list[ExamplePrediction[JmmluExample]]) -> dict[str, dict[str, int]]:
    # ---------------------------------------------------------
    # Count correct predictions per JMMLU subject for both the
    # terminal table and config.json summary.
    # ---------------------------------------------------------
    subject_counts: dict[str, dict[str, int]] = {}

    for row in rows:
        subject_count = subject_counts.setdefault(row.example.subject, {"correct": 0, "total": 0})
        subject_count["correct"] += int(row.correct)
        subject_count["total"] += 1

    return subject_counts


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


def resolve_output_dir(output_dir: str | None, model_source: str) -> Path:
    # ---------------------------------------------------------
    # Use an explicit output directory when provided. Otherwise,
    # create a timestamped directory under eval_results.
    # ---------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return build_output_dir(
        base_dir=DEFAULT_OUTPUT_DIR,
        output_dir=output_dir,
        model_source=model_source,
        timestamp=timestamp,
    )


def save_result(
    result: EvaluationResult,
    output_dir: Path,
    limit: int | None,
    subjects: list[str] | None,
) -> None:
    # ---------------------------------------------------------
    # Persist summary config and all per-example rows for one
    # evaluation run.
    # ---------------------------------------------------------
    config = {
        "model_source": result.model_source,
        "backend": result.backend,
        "dataset": result.dataset,
        "scoring_method": result.scoring_method,
        "device": result.device,
        "torch_dtype": result.torch_dtype,
        "limit": limit,
        "subjects": subjects,
        "overall": asdict(result.overall),
        "by_subject": [asdict(subject_result) for subject_result in result.by_subject],
    }
    rows = build_result_rows(rows=result.rows)
    save_evaluation_files(config=config, rows=rows, output_dir=output_dir)


def build_result_rows(rows: list[ExamplePrediction[JmmluExample]]) -> list[dict[str, object]]:
    # ---------------------------------------------------------
    # Convert JMMLU prediction records into CSV rows with question
    # text, choices, answers, predictions, and losses.
    # ---------------------------------------------------------
    return [
        {
            "index": row.index,
            "subject": row.example.subject,
            "question": row.example.question,
            "choice_A": row.example.choices[0],
            "choice_B": row.example.choices[1],
            "choice_C": row.example.choices[2],
            "choice_D": row.example.choices[3],
            "answer": row.example.answer,
            "prediction": row.prediction,
            "correct": row.correct,
            "loss_A": row.losses[0],
            "loss_B": row.losses[1],
            "loss_C": row.losses[2],
            "loss_D": row.losses[3],
        }
        for row in rows
    ]
