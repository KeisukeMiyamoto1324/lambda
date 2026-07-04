from dataclasses import dataclass

from src.eval.shared.models import ChoiceScorer


@dataclass(frozen=True)
class MultipleChoiceExample:
    group: str
    question: str
    choices: list[str]
    answer: str


def predict_choice(
    scorer: ChoiceScorer,
    prompt: str,
    answer_labels: tuple[str, ...],
    continuations: tuple[str, ...],
) -> str:
    # ---------------------------------------------------------
    # Score every candidate continuation and return the label
    # with the lowest language-model loss.
    # ---------------------------------------------------------
    losses = scorer.score_continuations(prompt=prompt, continuations=continuations)
    best_index = min(range(len(losses)), key=lambda index: losses[index])
    return answer_labels[best_index]
