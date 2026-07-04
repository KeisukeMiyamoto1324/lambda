from src.eval.jmmlu.dataset import ANSWER_LABELS
from src.eval.jmmlu.dataset import JmmluExample
from src.eval.shared.models import ChoiceScorer
from src.eval.shared.multiple_choice import predict_choice


ANSWER_CONTINUATIONS = tuple(f" {label}" for label in ANSWER_LABELS)


def build_prompt(example: JmmluExample) -> str:
    # ---------------------------------------------------------
    # Build the common MMLU-style zero-shot prompt with four
    # labeled choices and an answer label target.
    # ---------------------------------------------------------
    choices_text = "\n".join(
        f"{label}. {choice}"
        for label, choice in zip(ANSWER_LABELS, example.choices, strict=True)
    )
    return f"Question: {example.question}\n{choices_text}\nAnswer:"


def predict_answer(scorer: ChoiceScorer, example: JmmluExample) -> str:
    # ---------------------------------------------------------
    # Score each answer label and choose the label with the
    # lowest language-model loss.
    # ---------------------------------------------------------
    prompt = build_prompt(example=example)
    return predict_choice(
        scorer=scorer,
        prompt=prompt,
        answer_labels=ANSWER_LABELS,
        continuations=ANSWER_CONTINUATIONS,
    )
