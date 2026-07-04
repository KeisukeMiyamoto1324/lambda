from src.eval.jcommonsenseqa.dataset import ANSWER_LABELS
from src.eval.jcommonsenseqa.dataset import JCommonsenseQAExample
from src.eval.shared.models import ChoiceScorer
from src.eval.shared.multiple_choice import MultipleChoicePrediction
from src.eval.shared.multiple_choice import predict_choice


DATASET_INSTRUCTION = "質問と回答の選択肢を入力として受け取り、選択肢から回答を選択してください。なお、回答は選択肢の番号（例：0）でするものとします。 回答となる数値を返してください。"
ANSWER_PATTERN = "選択肢の番号または文字のみで回答してください"


def build_input(example: JCommonsenseQAExample) -> str:
    # ---------------------------------------------------------
    # Build the standard llm-jp-eval input text for this task.
    # Choices use numeric labels from 0 to 4.
    # ---------------------------------------------------------
    choices_text = ",".join(
        f"{label}.{choice}"
        for label, choice in zip(ANSWER_LABELS, example.choices, strict=True)
    )
    return f"質問：{example.question}\n選択肢：{choices_text}"


def build_prompt(example: JCommonsenseQAExample) -> str:
    # ---------------------------------------------------------
    # Build the zero-shot llm-jp-eval prompt with instruction,
    # answer format, input, and response prefix.
    # ---------------------------------------------------------
    return (
        "### 指示\n"
        f"{DATASET_INSTRUCTION}\n"
        "### 回答形式\n"
        f"{ANSWER_PATTERN}\n"
        "### 入力:\n"
        f"{build_input(example=example)}\n"
        "### 応答:\n"
    )


def predict_answer(scorer: ChoiceScorer, example: JCommonsenseQAExample) -> MultipleChoicePrediction:
    # ---------------------------------------------------------
    # Score numeric answer continuations and choose the label with
    # the lowest language-model loss.
    # ---------------------------------------------------------
    prompt = build_prompt(example=example)
    return predict_choice(
        scorer=scorer,
        prompt=prompt,
        answer_labels=ANSWER_LABELS,
        continuations=ANSWER_LABELS,
    )
