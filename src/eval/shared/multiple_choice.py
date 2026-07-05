from dataclasses import dataclass

from src.eval.shared.scorer_types import ChoiceScorer


@dataclass(frozen=True)
class MultipleChoiceExample:
    group: str
    question: str
    choices: list[str]
    answer: str


@dataclass(frozen=True)
class MultipleChoicePrediction:
    prediction: str
    losses: list[float]


def predict_choice(
    scorer: ChoiceScorer,
    prompt: str,
    answer_labels: tuple[str, ...],
    continuations: tuple[str, ...],
) -> MultipleChoicePrediction:
    # ---------------------------------------------------------
    # Score every candidate continuation and keep both the best
    # answer label and all per-choice losses.
    # ---------------------------------------------------------
    losses = scorer.score_continuations(prompt=prompt, continuations=continuations)
    best_index = min(range(len(losses)), key=lambda index: losses[index])
    return MultipleChoicePrediction(
        prediction=answer_labels[best_index],
        losses=losses,
    )
