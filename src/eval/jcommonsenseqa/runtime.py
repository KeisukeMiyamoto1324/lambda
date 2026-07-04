import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.eval.jcommonsenseqa.dataset import JCOMMONSENSEQA_CONFIG
from src.eval.jcommonsenseqa.dataset import JCOMMONSENSEQA_DATASET_ID
from src.eval.jcommonsenseqa.dataset import JCommonsenseQAExample
from src.eval.jcommonsenseqa.dataset import load_examples
from src.eval.jcommonsenseqa.scoring import predict_answer
from src.eval.shared.models import ChoiceScorer
from src.eval.shared.models import load_choice_scorer
from src.eval.shared.runtime import AccuracyResult
from src.eval.shared.runtime import count_correct_predictions
from src.eval.shared.runtime import save_json_result
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
    save_result(result=result, output_path=resolve_output_json(output_json=args.output_json))


def evaluate_examples(
    scorer: ChoiceScorer,
    examples: list[JCommonsenseQAExample],
    split: str,
) -> EvaluationResult:
    # ---------------------------------------------------------
    # Evaluate selected examples with shared exact-match logic and
    # return the benchmark-specific result shape.
    # ---------------------------------------------------------
    overall = count_correct_predictions(
        scorer=scorer,
        examples=examples,
        benchmark_name="JCommonsenseQA",
        predict_answer=predict_answer,
    )

    return build_evaluation_result(
        scorer=scorer,
        overall=overall,
        split=split,
    )


def build_evaluation_result(
    scorer: ChoiceScorer,
    overall: AccuracyResult,
    split: str,
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
    save_json_result(result=result, output_path=output_path)
