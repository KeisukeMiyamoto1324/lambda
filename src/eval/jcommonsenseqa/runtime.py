import argparse
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.eval.jcommonsenseqa.dataset import JCOMMONSENSEQA_CONFIG
from src.eval.jcommonsenseqa.dataset import JCOMMONSENSEQA_DATASET_ID
from src.eval.jcommonsenseqa.dataset import JCommonsenseQAExample
from src.eval.jcommonsenseqa.dataset import load_examples
from src.eval.jcommonsenseqa.scoring import predict_answer
from src.eval.shared.scorer_loader import load_choice_scorer
from src.eval.shared.scorer_types import ChoiceScorer
from src.eval.shared.runtime import AccuracyResult
from src.eval.shared.runtime import ExamplePrediction
from src.eval.shared.runtime import build_output_dir
from src.eval.shared.runtime import collect_predictions
from src.eval.shared.runtime import save_evaluation_files
from src.eval.shared.runtime import select_examples
from src.shared.console import console


DEFAULT_OUTPUT_DIR = Path("eval_results/jcommonsenseqa")


@dataclass(frozen=True)
class EvaluationResult:
    model_source: str
    backend: str
    dataset: str
    config: str
    split: str
    scoring_method: str
    device: str
    torch_dtype: str
    overall: AccuracyResult
    rows: list[ExamplePrediction[JCommonsenseQAExample]]


def run_evaluation(args: argparse.Namespace) -> None:
    # ---------------------------------------------------------
    # Load the selected model scorer and JCommonsenseQA examples,
    # then run evaluation and save the final metrics.
    # ---------------------------------------------------------
    scorer = load_choice_scorer(
        model_source=args.model,
        backend=args.backend,
        torch_dtype_name=args.torch_dtype,
        trust_remote_code=args.trust_remote_code,
    )
    examples = load_examples(split=args.split)
    selected_examples = select_examples(
        examples=examples,
        limit=args.limit,
        benchmark_name="JCommonsenseQA",
    )

    result = evaluate_examples(
        scorer=scorer,
        examples=selected_examples,
        split=args.split,
    )
    render_result(result=result)
    output_dir = resolve_output_dir(output_dir=args.output_dir, model_source=args.model)
    save_result(
        result=result,
        output_dir=output_dir,
        limit=args.limit,
    )


def evaluate_examples(
    scorer: ChoiceScorer,
    examples: list[JCommonsenseQAExample],
    split: str,
) -> EvaluationResult:
    # ---------------------------------------------------------
    # Evaluate selected examples with shared exact-match logic and
    # return the benchmark-specific result shape.
    # ---------------------------------------------------------
    overall, rows = collect_predictions(
        scorer=scorer,
        examples=examples,
        benchmark_name="JCommonsenseQA",
        predict_answer=predict_answer,
    )

    return build_evaluation_result(
        scorer=scorer,
        overall=overall,
        split=split,
        rows=rows,
    )


def build_evaluation_result(
    scorer: ChoiceScorer,
    overall: AccuracyResult,
    split: str,
    rows: list[ExamplePrediction[JCommonsenseQAExample]],
) -> EvaluationResult:
    # ---------------------------------------------------------
    # Convert raw counters into a serializable result dataclass
    # for terminal rendering and JSON output.
    # ---------------------------------------------------------
    return EvaluationResult(
        model_source=scorer.model_source,
        backend=scorer.backend,
        dataset=JCOMMONSENSEQA_DATASET_ID,
        config=JCOMMONSENSEQA_CONFIG,
        split=split,
        scoring_method="zero_shot_llm_jp_eval_numeric_label_log_likelihood",
        device=scorer.device_name,
        torch_dtype=scorer.torch_dtype_name,
        overall=overall,
        rows=rows,
    )


def render_result(result: EvaluationResult) -> None:
    # ---------------------------------------------------------
    # Print overall metrics with Rich console output so terminal
    # results stay easy to scan.
    # ---------------------------------------------------------
    console.print("[bold cyan]JCommonsenseQA result[/bold cyan]")
    console.print(f"model: {result.model_source}")
    console.print(f"backend: {result.backend}")
    console.print(f"split: {result.split}")
    console.print(f"accuracy: {result.overall.accuracy:.4f}")
    console.print(f"correct: {result.overall.correct}")
    console.print(f"total: {result.overall.total}")


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


def save_result(result: EvaluationResult, output_dir: Path, limit: int | None) -> None:
    # ---------------------------------------------------------
    # Persist summary config and all per-example rows for one
    # evaluation run.
    # ---------------------------------------------------------
    config = {
        "model_source": result.model_source,
        "backend": result.backend,
        "dataset": result.dataset,
        "config": result.config,
        "split": result.split,
        "scoring_method": result.scoring_method,
        "device": result.device,
        "torch_dtype": result.torch_dtype,
        "limit": limit,
        "overall": asdict(result.overall),
    }
    rows = build_result_rows(rows=result.rows)
    save_evaluation_files(config=config, rows=rows, output_dir=output_dir)


def build_result_rows(rows: list[ExamplePrediction[JCommonsenseQAExample]]) -> list[dict[str, object]]:
    # ---------------------------------------------------------
    # Convert JCommonsenseQA prediction records into CSV rows with
    # question text, choices, answers, predictions, and losses.
    # ---------------------------------------------------------
    return [
        {
            "index": row.index,
            "q_id": row.example.q_id,
            "question": row.example.question,
            "choice_0": row.example.choices[0],
            "choice_1": row.example.choices[1],
            "choice_2": row.example.choices[2],
            "choice_3": row.example.choices[3],
            "choice_4": row.example.choices[4],
            "answer": row.example.answer,
            "prediction": row.prediction,
            "correct": row.correct,
            "loss_0": row.losses[0],
            "loss_1": row.losses[1],
            "loss_2": row.losses[2],
            "loss_3": row.losses[3],
            "loss_4": row.losses[4],
        }
        for row in rows
    ]
