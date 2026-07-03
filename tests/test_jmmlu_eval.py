import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn

from src.eval.jmmlu.cli import parse_args
from src.eval.jmmlu.dataset import ANSWER_LABELS
from src.eval.jmmlu.dataset import JmmluExample
from src.eval.jmmlu.dataset import load_examples
from src.eval.jmmlu.models import NativeChoiceScorer
from src.eval.jmmlu.models import TransformersChoiceScorer
from src.eval.jmmlu.models import build_hf_labels
from src.eval.jmmlu.models import compute_row_losses
from src.eval.jmmlu.models import resolve_backend
from src.eval.jmmlu.models import score_native_answer_label
from src.eval.jmmlu.runtime import evaluate_examples
from src.eval.jmmlu.runtime import save_result
from src.eval.jmmlu.scoring import build_prompt


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


class FakeTransformersTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict[str, list[int]]:
        # ---------------------------------------------------------
        # Encode characters into small ids for deterministic
        # Transformers scorer tests.
        # ---------------------------------------------------------
        prefix = [1] if add_special_tokens else []
        return {"input_ids": [*prefix, *[ord(character) for character in text]]}


class FakeTransformersOutput:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class FakeTransformersModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.probe = nn.Parameter(torch.zeros(1))
        self.labels: torch.Tensor | None = None

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> FakeTransformersOutput:
        # ---------------------------------------------------------
        # Create logits that make lower token ids less costly so
        # answer A is preferred over B/C/D.
        # ---------------------------------------------------------
        del attention_mask
        vocab_size = 128
        logits = torch.zeros((*input_ids.shape, vocab_size), dtype=torch.float32, device=input_ids.device)
        logits[:, :, ord("A")] = 4.0
        logits[:, :, ord("B")] = 3.0
        logits[:, :, ord("C")] = 2.0
        logits[:, :, ord("D")] = 1.0
        return FakeTransformersOutput(logits=logits)


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

    def test_parse_args_accepts_model_and_backend(self) -> None:
        # ---------------------------------------------------------
        # Parse the simplified JMMLU CLI without keeping the old
        # model directory flag.
        # ---------------------------------------------------------
        with patch("sys.argv", ["evaluate.py", "--model", "Qwen/Qwen3-0.6B", "--backend", "hf"]):
            args = parse_args()

        self.assertEqual(args.model, "Qwen/Qwen3-0.6B")
        self.assertEqual(args.backend, "hf")

    def test_resolve_backend_detects_native_model_artifacts(self) -> None:
        # ---------------------------------------------------------
        # Auto-select native only when local PyTorch model artifacts
        # are present in the model directory.
        # ---------------------------------------------------------
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            (model_dir / "model.pth").write_bytes(b"")
            (model_dir / "model_config.json").write_text("{}", encoding="utf-8")

            backend = resolve_backend(model_source=str(model_dir), backend="auto")

        self.assertEqual(backend, "native")
        self.assertEqual(resolve_backend(model_source="Qwen/Qwen3-0.6B", backend="auto"), "hf")

    def test_resolve_backend_rejects_missing_local_path(self) -> None:
        # ---------------------------------------------------------
        # Reject path-like local model sources before trying to
        # resolve them as Hugging Face repository ids.
        # ---------------------------------------------------------
        with self.assertRaises(FileNotFoundError):
            resolve_backend(model_source="models/missing-model", backend="auto")

    def test_score_native_answer_label_masks_prompt_tokens(self) -> None:
        # ---------------------------------------------------------
        # Keep prompt labels masked with pad_token_id and score
        # only the answer label suffix.
        # ---------------------------------------------------------
        model = FakeModel()
        tokenizer = FakeTokenizer()

        loss = score_native_answer_label(
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

    def test_build_hf_labels_masks_prompt_and_padding_tokens(self) -> None:
        # ---------------------------------------------------------
        # Keep only answer suffix positions visible to the HF loss
        # calculation.
        # ---------------------------------------------------------
        labels = build_hf_labels(
            full_token_ids=[[1, 10, 20, 30], [1, 10, 21]],
            prompt_len=2,
            max_len=4,
            pad_token_id=0,
        )

        self.assertEqual(labels, [[-100, 20, 30, -100], [-100, 21, -100, -100]])

    def test_compute_row_losses_returns_per_choice_losses(self) -> None:
        # ---------------------------------------------------------
        # Average unmasked token losses per row so batched HF
        # scoring can still compare each answer independently.
        # ---------------------------------------------------------
        logits = torch.zeros((2, 3, 4), dtype=torch.float32)
        logits[0, 1, 1] = 5.0
        logits[1, 1, 2] = 5.0
        labels = torch.tensor([[-100, 1, -100], [-100, 1, -100]])

        losses = compute_row_losses(logits=logits, labels=labels)

        self.assertLess(losses[0], losses[1])

    def test_transformers_choice_scorer_batches_answer_labels(self) -> None:
        # ---------------------------------------------------------
        # Score answer labels through the generic HF scorer without
        # using a real remote model.
        # ---------------------------------------------------------
        scorer = TransformersChoiceScorer(
            model=FakeTransformersModel(),
            tokenizer=FakeTransformersTokenizer(),
            device=torch.device("cpu"),
            model_source="fake/hf",
            torch_dtype_name="auto",
        )

        losses = scorer.score_answer_labels(prompt="Question: test\nAnswer:", answer_labels=ANSWER_LABELS)

        self.assertEqual(len(losses), 4)
        self.assertEqual(min(range(len(losses)), key=lambda index: losses[index]), 0)

    def test_evaluate_examples_returns_overall_and_subject_accuracy(self) -> None:
        # ---------------------------------------------------------
        # Aggregate deterministic fake predictions into overall
        # and subject-level result objects.
        # ---------------------------------------------------------
        model = FakeModel()
        tokenizer = FakeTokenizer()
        scorer = NativeChoiceScorer(
            model=model,
            tokenizer=tokenizer,
            max_seq_len=128,
            pad_token_id=0,
            bos_token_id=1,
            device=torch.device("cpu"),
            model_source="models/test",
            torch_dtype_name="auto",
        )
        result = evaluate_examples(
            scorer=scorer,
            examples=[
                JmmluExample("s1", "q1", ["a", "b", "c", "d"], "A"),
                JmmluExample("s1", "q2", ["a", "b", "c", "d"], "B"),
            ],
        )

        self.assertEqual(ANSWER_LABELS, ("A", "B", "C", "D"))
        self.assertEqual(result.backend, "native")
        self.assertEqual(result.overall.correct, 1)
        self.assertEqual(result.overall.total, 2)
        self.assertEqual(result.overall.accuracy, 0.5)
        self.assertEqual(result.by_subject[0].subject, "s1")

    def test_save_result_writes_json(self) -> None:
        # ---------------------------------------------------------
        # Save JSON summaries to the requested path and create
        # missing parent directories.
        # ---------------------------------------------------------
        scorer = NativeChoiceScorer(
            model=FakeModel(),
            tokenizer=FakeTokenizer(),
            max_seq_len=128,
            pad_token_id=0,
            bos_token_id=1,
            device=torch.device("cpu"),
            model_source="models/test",
            torch_dtype_name="auto",
        )
        result = evaluate_examples(
            scorer=scorer,
            examples=[JmmluExample("s1", "q1", ["a", "b", "c", "d"], "A")],
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
