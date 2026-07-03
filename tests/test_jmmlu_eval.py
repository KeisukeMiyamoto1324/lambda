import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn

from src.eval.jmmlu.dataset import ANSWER_LABELS
from src.eval.jmmlu.dataset import JmmluExample
from src.eval.jmmlu.dataset import load_examples
from src.eval.jmmlu.runtime import evaluate_examples
from src.eval.jmmlu.runtime import save_result
from src.eval.jmmlu.scoring import build_prompt
from src.eval.jmmlu.scoring import score_answer_label


class FakeTokenizer:
    pad_token = "|<pad>|"
    bos_token = "|<bos>|"

    def token_to_id(self, token: str) -> int:
        # ---------------------------------------------------------
        # Return stable special-token ids for scoring tests.
        # ---------------------------------------------------------
        token_ids = {
            self.pad_token: 0,
            self.bos_token: 1,
        }
        return token_ids[token]

    def tokenize(self, sentence: str) -> list[int]:
        # ---------------------------------------------------------
        # Encode characters into stable ids so tests can inspect
        # which label tokens are scored.
        # ---------------------------------------------------------
        return [ord(character) for character in sentence]


class FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # ---------------------------------------------------------
        # Keep a parameter for normal torch module behavior and
        # record every label tensor passed by the evaluator.
        # ---------------------------------------------------------
        self.probe = nn.Parameter(torch.zeros(1))
        self.labels: list[list[int]] = []

    def compute_chunked_loss(self, input_tokens: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # ---------------------------------------------------------
        # Prefer answer A by giving its label token the lowest fake
        # loss. This keeps evaluation deterministic.
        # ---------------------------------------------------------
        del input_tokens
        label_ids = [int(token_id) for token_id in labels[0].tolist()]
        self.labels.append(label_ids)
        scored_ids = [token_id for token_id in label_ids if token_id != 0]
        return torch.tensor(float(scored_ids[-1] - ord("A")))


class JmmluEvalTest(unittest.TestCase):
    def test_build_prompt_uses_mmlu_answer_label_format(self) -> None:
        # ---------------------------------------------------------
        # Build a zero-shot MMLU-style prompt with four labeled
        # choices and an answer label target.
        # ---------------------------------------------------------
        example = JmmluExample(
            subject="sample",
            question="質問文",
            choices=["選択肢1", "選択肢2", "選択肢3", "選択肢4"],
            answer="A",
        )

        prompt = build_prompt(example=example)

        self.assertEqual(
            prompt,
            "Question: 質問文\nA. 選択肢1\nB. 選択肢2\nC. 選択肢3\nD. 選択肢4\nAnswer:",
        )

    def test_load_examples_reads_jmmlu_zip_csv(self) -> None:
        # ---------------------------------------------------------
        # Read only official JMMLU/test CSV entries and convert
        # answer columns into the evaluator example shape.
        # ---------------------------------------------------------
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "JMMLU.zip"

            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(
                    "JMMLU/test/sample_subject.csv",
                    "question,A,B,C,D,answer\n質問,甲,乙,丙,丁,B\n",
                )
                archive.writestr("__MACOSX/JMMLU/test/._sample_subject.csv", "")

            examples = load_examples(archive_path=archive_path, subjects=["sample_subject"])

        self.assertEqual(
            examples,
            [
                JmmluExample(
                    subject="sample_subject",
                    question="質問",
                    choices=["甲", "乙", "丙", "丁"],
                    answer="B",
                )
            ],
        )

    def test_score_answer_label_masks_prompt_tokens(self) -> None:
        # ---------------------------------------------------------
        # Keep prompt labels masked with pad_token_id and score
        # only the answer label suffix.
        # ---------------------------------------------------------
        model = FakeModel()
        tokenizer = FakeTokenizer()

        loss = score_answer_label(
            model=model,
            tokenizer=tokenizer,
            prompt="Question: test\nAnswer:",
            answer_label="A",
            device=torch.device("cpu"),
            pad_token_id=0,
            bos_token_id=1,
            max_seq_len=128,
        )

        self.assertEqual(loss, 0.0)
        self.assertEqual(model.labels[-1][-1], ord("A"))
        self.assertTrue(all(token_id == 0 for token_id in model.labels[-1][:-2]))

    def test_evaluate_examples_returns_overall_and_subject_accuracy(self) -> None:
        # ---------------------------------------------------------
        # Aggregate deterministic fake predictions into overall
        # and subject-level result objects.
        # ---------------------------------------------------------
        result = evaluate_examples(
            model=FakeModel(),
            tokenizer=FakeTokenizer(),
            examples=[
                JmmluExample("s1", "q1", ["a", "b", "c", "d"], "A"),
                JmmluExample("s1", "q2", ["a", "b", "c", "d"], "B"),
            ],
            model_source="models/test",
            device=torch.device("cpu"),
            torch_dtype="auto",
            max_seq_len=128,
        )

        self.assertEqual(ANSWER_LABELS, ("A", "B", "C", "D"))
        self.assertEqual(result.overall.correct, 1)
        self.assertEqual(result.overall.total, 2)
        self.assertEqual(result.overall.accuracy, 0.5)
        self.assertEqual(result.by_subject[0].subject, "s1")

    def test_save_result_writes_json(self) -> None:
        # ---------------------------------------------------------
        # Save JSON summaries to the requested path and create
        # missing parent directories.
        # ---------------------------------------------------------
        result = evaluate_examples(
            model=FakeModel(),
            tokenizer=FakeTokenizer(),
            examples=[JmmluExample("s1", "q1", ["a", "b", "c", "d"], "A")],
            model_source="models/test",
            device=torch.device("cpu"),
            torch_dtype="auto",
            max_seq_len=128,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "nested" / "result.json"
            with patch("src.eval.jmmlu.runtime.console.print"):
                save_result(result=result, output_path=output_path)

            text = output_path.read_text(encoding="utf-8")

        self.assertIn('"dataset": "nlp-waseda/JMMLU"', text)
        self.assertIn('"accuracy": 1.0', text)


if __name__ == "__main__":
    unittest.main()
