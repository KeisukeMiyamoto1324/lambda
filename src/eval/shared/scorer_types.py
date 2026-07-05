from dataclasses import dataclass
from typing import Protocol


class ChoiceScorer(Protocol):
    backend: str
    model_source: str
    device_name: str
    torch_dtype_name: str

    def score_continuations(self, prompt: str, continuations: tuple[str, ...]) -> list[float]:
        ...

    def score_answer_labels(self, prompt: str, answer_labels: tuple[str, ...]) -> list[float]:
        ...


@dataclass(frozen=True)
class TextScore:
    loss_sum: float
    token_count: int


class TextScorer(ChoiceScorer, Protocol):
    def score_text(self, text: str) -> TextScore:
        ...
