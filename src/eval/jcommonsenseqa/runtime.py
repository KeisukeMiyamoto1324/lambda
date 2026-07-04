import argparse
import json
from dataclasses import asdict
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
from src.shared.console import console
from src.shared.console import progress_manager


DEFAULT_OUTPUT_DIR = Path("eval_results/jcommonsenseqa")


@dataclass(frozen=True)
class AccuracyResult:
    accuracy: float
    correct: int
    total: int


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
    selected_examples = examples if args.limit is None else examples[: args.limit]

    if not selected_examples:
        raise ValueError("No JCommonsenseQA examples were selected")

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
    # Evaluate selected examples and track overall exact-match
    # accuracy for numeric answer labels.
    # ---------------------------------------------------------
    correct = 0
    task_id = progress_manager.add_task(description="JCommonsenseQA", total=len(examples))

    try:
        for index, example in enumerate(examples, start=1):
            prediction = predict_answer(
                scorer=scorer,
                example=example,
            )
            correct += int(prediction == example.answer)
            progress_manager.update(
                task_id=task_id,
                advance=1,
                metrics=f"accuracy={correct / index:.4f}",
            )
    finally:
        progress_manager.finish_task(task_id=task_id)

    return build_evaluation_result(
        scorer=scorer,
        correct=correct,
        total=len(examples),
        split=split,
    )


def build_evaluation_result(
    scorer: ChoiceScorer,
    correct: int,
    total: int,
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
        overall=AccuracyResult(
            accuracy=correct / total,
            correct=correct,
            total=total,
        ),
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(asdict(result), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    console.print(f"[cyan]saved json[/cyan] {output_path}")
