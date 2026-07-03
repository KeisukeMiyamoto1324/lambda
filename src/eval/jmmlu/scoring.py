from src.eval.jmmlu.dataset import ANSWER_LABELS
from src.eval.jmmlu.dataset import JmmluExample
from src.eval.jmmlu.models import ChoiceScorer


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
    losses = scorer.score_answer_labels(prompt=prompt, answer_labels=ANSWER_LABELS)
    best_index = min(range(len(losses)), key=lambda index: losses[index])
    return ANSWER_LABELS[best_index]
