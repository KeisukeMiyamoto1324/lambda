import argparse
import math
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.eval.jsquad_perplexity.dataset import JSQUAD_CONFIG
from src.eval.jsquad_perplexity.dataset import JSQUAD_DATASET_ID
from src.eval.jsquad_perplexity.dataset import JSQUAD_SPLIT
from src.eval.jsquad_perplexity.dataset import JSQuADContext
from src.eval.jsquad_perplexity.dataset import load_contexts
from src.eval.shared.scorer_loader import load_choice_scorer
from src.eval.shared.scorer_types import TextScore
from src.eval.shared.scorer_types import TextScorer
from src.eval.shared.runtime import build_output_dir
from src.eval.shared.runtime import save_evaluation_files
from src.eval.shared.runtime import select_examples
from src.shared.console import console
from src.shared.console import progress_manager


DEFAULT_OUTPUT_DIR = Path("eval_results/jsquad_perplexity")


@dataclass(frozen=True)
class PerplexityResult:
    loss: float
    perplexity: float
    loss_sum: float
    token_count: int
    total: int


@dataclass(frozen=True)
class ContextPrediction:
    index: int
    example: JSQuADContext
    score: TextScore
    loss: float
    perplexity: float


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
    overall: PerplexityResult
    rows: list[ContextPrediction]


def run_evaluation(args: argparse.Namespace) -> None:
    # ---------------------------------------------------------
    # Load the selected model scorer and unique JSQuAD contexts,
    # then run perplexity evaluation and save the final metrics.
    # ---------------------------------------------------------
    scorer = load_choice_scorer(
        model_source=args.model,
        backend=args.backend,
        torch_dtype_name=args.torch_dtype,
        trust_remote_code=args.trust_remote_code,
    )
    contexts = load_contexts()
    selected_contexts = select_examples(
        examples=contexts,
        limit=args.limit,
        benchmark_name="JSQuAD perplexity",
    )

    result = evaluate_contexts(
        scorer=scorer,
        contexts=selected_contexts,
    )
    render_result(result=result)
    output_dir = resolve_output_dir(output_dir=args.output_dir, model_source=args.model)
    save_result(
        result=result,
        output_dir=output_dir,
        limit=args.limit,
    )


def evaluate_contexts(scorer: TextScorer, contexts: list[JSQuADContext]) -> EvaluationResult:
    # ---------------------------------------------------------
    # Score every unique context and aggregate token-weighted
    # perplexity across the full selected corpus.
    # ---------------------------------------------------------
    rows = collect_context_scores(scorer=scorer, contexts=contexts)
    overall = build_perplexity_result(rows=rows)
    return EvaluationResult(
        model_source=scorer.model_source,
        backend=scorer.backend,
        dataset=JSQUAD_DATASET_ID,
        config=JSQUAD_CONFIG,
        split=JSQUAD_SPLIT,
        scoring_method="unique_context_causal_lm_perplexity",
        device=scorer.device_name,
        torch_dtype=scorer.torch_dtype_name,
        overall=overall,
        rows=rows,
    )


def collect_context_scores(scorer: TextScorer, contexts: list[JSQuADContext]) -> list[ContextPrediction]:
    # ---------------------------------------------------------
    # Run context scoring with progress metrics based on the
    # accumulated token-weighted loss.
    # ---------------------------------------------------------
    rows: list[ContextPrediction] = []
    loss_sum = 0.0
    token_count = 0
    task_id = progress_manager.add_task(description="JSQuAD perplexity", total=len(contexts))

    try:
        for index, context in enumerate(contexts, start=1):
            score = scorer.score_text(text=context.context)
            loss = score.loss_sum / score.token_count
            prediction = ContextPrediction(
                index=index,
                example=context,
                score=score,
                loss=loss,
                perplexity=math.exp(loss),
            )
            rows.append(prediction)
            loss_sum += score.loss_sum
            token_count += score.token_count
            progress_manager.update(
                task_id=task_id,
                advance=1,
                metrics=f"ppl={math.exp(loss_sum / token_count):.4f}",
            )
    finally:
        progress_manager.finish_task(task_id=task_id)

    return rows


def build_perplexity_result(rows: list[ContextPrediction]) -> PerplexityResult:
    # ---------------------------------------------------------
    # Aggregate context scores with token weighting so short and
    # long contexts contribute by their number of target tokens.
    # ---------------------------------------------------------
    loss_sum = sum(row.score.loss_sum for row in rows)
    token_count = sum(row.score.token_count for row in rows)
    loss = loss_sum / token_count
    return PerplexityResult(
        loss=loss,
        perplexity=math.exp(loss),
        loss_sum=loss_sum,
        token_count=token_count,
        total=len(rows),
    )


def render_result(result: EvaluationResult) -> None:
    # ---------------------------------------------------------
    # Print overall metrics with Rich console output so terminal
    # results stay easy to scan.
    # ---------------------------------------------------------
    console.print("[bold cyan]JSQuAD perplexity result[/bold cyan]")
    console.print(f"model: {result.model_source}")
    console.print(f"backend: {result.backend}")
    console.print(f"split: {result.split}")
    console.print(f"loss: {result.overall.loss:.4f}")
    console.print(f"perplexity: {result.overall.perplexity:.4f}")
    console.print(f"token_count: {result.overall.token_count}")
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
    # Persist summary config and all per-context rows for one
    # perplexity evaluation run.
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


def build_result_rows(rows: list[ContextPrediction]) -> list[dict[str, object]]:
    # ---------------------------------------------------------
    # Convert JSQuAD context score records into CSV rows with
    # per-context loss, perplexity, and token counts.
    # ---------------------------------------------------------
    return [
        {
            "index": row.index,
            "context_id": row.example.context_id,
            "token_count": row.score.token_count,
            "loss": row.loss,
            "perplexity": row.perplexity,
            "context": row.example.context,
        }
        for row in rows
    ]
