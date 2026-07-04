import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.eval.jcommonsenseqa.cli import parse_args
from src.eval.jcommonsenseqa.dataset import ANSWER_LABELS
from src.eval.jcommonsenseqa.dataset import JCOMMONSENSEQA_DATASET_ID
from src.eval.jcommonsenseqa.dataset import JCommonsenseQAExample
from src.eval.jcommonsenseqa.dataset import build_example
from src.eval.jcommonsenseqa.runtime import evaluate_examples
from src.eval.jcommonsenseqa.runtime import save_result
from src.eval.jcommonsenseqa.scoring import build_input
from src.eval.jcommonsenseqa.scoring import build_prompt
from src.eval.jcommonsenseqa.scoring import predict_answer


class FakeChoiceScorer:
    backend = "fake"
    model_source = "models/test"
    device_name = "cpu"
    torch_dtype_name = "auto"

    def score_continuations(self, prompt: str, continuations: tuple[str, ...]) -> list[float]:
        # ---------------------------------------------------------
        # Prefer answer 2 so prediction and evaluation are
        # deterministic without loading a real model.
        # ---------------------------------------------------------
        del prompt
        return [0.0 if continuation == "2" else 1.0 for continuation in continuations]

    def score_answer_labels(self, prompt: str, answer_labels: tuple[str, ...]) -> list[float]:
        # ---------------------------------------------------------
        # Keep the fake scorer compatible with the shared protocol.
        # ---------------------------------------------------------
        return self.score_continuations(prompt=prompt, continuations=answer_labels)


class JCommonsenseQAEvalTest(unittest.TestCase):
    def test_build_example_converts_hugging_face_row(self) -> None:
        # ---------------------------------------------------------
        # Convert the official row shape into the evaluator example
        # with numeric answer labels.
        # ---------------------------------------------------------
        row = {
            "q_id": 1,
            "question": "質問",
            "choice0": "a",
            "choice1": "b",
            "choice2": "c",
            "choice3": "d",
            "choice4": "e",
            "label": 2,
        }

        example = build_example(row=row)

        self.assertEqual(example.q_id, 1)
        self.assertEqual(example.choices, ["a", "b", "c", "d", "e"])
        self.assertEqual(example.answer, "2")

    def test_build_prompt_uses_llm_jp_eval_format(self) -> None:
        # ---------------------------------------------------------
        # Build the standard JCommonsenseQA input and response
        # prefix with numeric choice labels.
        # ---------------------------------------------------------
        example = JCommonsenseQAExample(
            q_id=0,
            question="主に子ども向けのものはどれ？",
            choices=["世界", "写真集", "絵本", "論文", "図鑑"],
            answer="2",
        )

        input_text = build_input(example=example)
        prompt = build_prompt(example=example)

        self.assertEqual(input_text, "質問：主に子ども向けのものはどれ？\n選択肢：0.世界,1.写真集,2.絵本,3.論文,4.図鑑")
        self.assertIn("### 指示\n", prompt)
        self.assertIn("### 回答形式\n", prompt)
        self.assertTrue(prompt.endswith("### 応答:\n"))

    def test_predict_answer_scores_numeric_labels(self) -> None:
        # ---------------------------------------------------------
        # Score numeric labels 0 through 4 and return the best
        # continuation label.
        # ---------------------------------------------------------
        example = JCommonsenseQAExample(
            q_id=0,
            question="質問",
            choices=["a", "b", "c", "d", "e"],
            answer="2",
        )

        prediction = predict_answer(scorer=FakeChoiceScorer(), example=example)

        self.assertEqual(ANSWER_LABELS, ("0", "1", "2", "3", "4"))
        self.assertEqual(prediction, "2")

    def test_parse_args_defaults_to_validation_split(self) -> None:
        # ---------------------------------------------------------
        # Use validation as the default public evaluation split.
        # ---------------------------------------------------------
        with patch("sys.argv", ["evaluate.py", "--model", "Qwen/Qwen3-0.6B", "--backend", "hf"]):
            args = parse_args()

        self.assertEqual(args.split, "validation")
        self.assertEqual(args.model, "Qwen/Qwen3-0.6B")

    def test_evaluate_examples_and_save_result(self) -> None:
        # ---------------------------------------------------------
        # Aggregate exact-match accuracy and write the result JSON
        # to the requested path.
        # ---------------------------------------------------------
        examples = [
            JCommonsenseQAExample(0, "q1", ["a", "b", "c", "d", "e"], "2"),
            JCommonsenseQAExample(1, "q2", ["a", "b", "c", "d", "e"], "1"),
        ]
        result = evaluate_examples(
            scorer=FakeChoiceScorer(),
            examples=examples,
            split="validation",
        )

        self.assertEqual(result.dataset, JCOMMONSENSEQA_DATASET_ID)
        self.assertEqual(result.overall.correct, 1)
        self.assertEqual(result.overall.total, 2)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "nested" / "result.json"
            with patch("src.eval.jcommonsenseqa.runtime.console.print"):
                save_result(result=result, output_path=output_path)

            text = output_path.read_text(encoding="utf-8")

        self.assertIn('"dataset": "sbintuitions/JCommonsenseQA"', text)
        self.assertIn('"split": "validation"', text)
        self.assertIn('"accuracy": 0.5', text)


if __name__ == "__main__":
    unittest.main()
