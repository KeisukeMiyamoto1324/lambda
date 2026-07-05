import tempfile
import unittest
import zipfile
import csv
import json
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn

from src.eval.jmmlu.cli import parse_args
from src.eval.jmmlu.dataset import ANSWER_LABELS
from src.eval.jmmlu.dataset import JmmluExample
from src.eval.jmmlu.dataset import load_examples
from src.eval.shared.hf_scorer import TransformersChoiceScorer
from src.eval.shared.labeling import build_continuation_labels
from src.eval.shared.labeling import build_hf_labels
from src.eval.shared.losses import compute_row_losses
from src.eval.shared.native_scorer import NativeChoiceScorer
from src.eval.shared.native_scorer import score_native_answer_label
from src.eval.shared.scorer_loader import resolve_backend
from src.eval.jmmlu.runtime import evaluate_examples
from src.eval.jmmlu.runtime import save_result
from src.eval.jmmlu.scoring import build_prompt


class FakeTokenizer:
    pad_token = "|<pad>|"
    bos_token = "|<bos>|"

    def __init__(self) -> None:
        # ---------------------------------------------------------
        # Expose a tokenizer-like object for offset-aware native
        # continuation scoring tests.
        # ---------------------------------------------------------
        self.tokenizer = self

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

    def encode(self, sentence: str) -> "FakeEncoding":
        # ---------------------------------------------------------
        # Return ids and character offsets like the tokenizer used
        # by the native scoring path.
        # ---------------------------------------------------------
        return FakeEncoding(
            ids=self.tokenize(sentence),
            offsets=[(index, index + 1) for index in range(len(sentence))],
        )


class BoundaryMergingTokenizer(FakeTokenizer):
    def encode(self, sentence: str) -> "FakeEncoding":
        # ---------------------------------------------------------
        # Merge the prompt-final colon and answer label into one
        # token to verify boundary-aware label selection.
        # ---------------------------------------------------------
        if sentence.endswith(": A"):
            prefix = sentence[:-3]
            return FakeEncoding(
                ids=[*[ord(character) for character in prefix], 90],
                offsets=[
                    *[(index, index + 1) for index in range(len(prefix))],
                    (len(prefix), len(sentence)),
                ],
            )

        if sentence.endswith(":A"):
            prefix = sentence[:-2]
            return FakeEncoding(
                ids=[*[ord(character) for character in prefix], 90],
                offsets=[
                    *[(index, index + 1) for index in range(len(prefix))],
                    (len(prefix), len(sentence)),
                ],
            )

        return super().encode(sentence=sentence)


class FakeEncoding:
    def __init__(self, ids: list[int], offsets: list[tuple[int, int]]) -> None:
        self.ids = ids
        self.offsets = offsets


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

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = True,
        return_offsets_mapping: bool = False,
    ) -> dict[str, list[int] | list[tuple[int, int]]]:
        # ---------------------------------------------------------
        # Encode characters into small ids for deterministic
        # Transformers scorer tests.
        # ---------------------------------------------------------
        prefix = [1] if add_special_tokens else []
        prefix_offsets = [(0, 0)] if add_special_tokens else []
        encoded = {
            "input_ids": [*prefix, *[ord(character) for character in text]],
            "offset_mapping": [*prefix_offsets, *[(index, index + 1) for index in range(len(text))]],
        }

        if return_offsets_mapping:
            return encoded

        return {"input_ids": encoded["input_ids"]}


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


class BoundaryAwareTransformersTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = True,
        return_offsets_mapping: bool = False,
    ) -> dict[str, list[int] | list[tuple[int, int]]]:
        # ---------------------------------------------------------
        # Encode standalone answer labels differently from labels
        # attached to a prompt, like boundary-sensitive tokenizers.
        # ---------------------------------------------------------
        prefix = [1] if add_special_tokens else []
        prefix_offsets = [(0, 0)] if add_special_tokens else []
        offset = 20 if not add_special_tokens else 0
        encoded = {
            "input_ids": [*prefix, *[ord(character) + offset for character in text]],
            "offset_mapping": [*prefix_offsets, *[(index, index + 1) for index in range(len(text))]],
        }

        if return_offsets_mapping:
            return encoded

        return {"input_ids": encoded["input_ids"]}


class BoundaryMergingTransformersTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = True,
        return_offsets_mapping: bool = False,
    ) -> dict[str, list[int] | list[tuple[int, int]]]:
        # ---------------------------------------------------------
        # Merge the prompt-final colon and answer label into one
        # token to verify offset-based label selection.
        # ---------------------------------------------------------
        prefix_ids = [1] if add_special_tokens else []
        prefix_offsets = [(0, 0)] if add_special_tokens else []

        if text.endswith(":A"):
            body = text[:-2]
            encoded = {
                "input_ids": [*prefix_ids, *[ord(character) for character in body], 90],
                "offset_mapping": [
                    *prefix_offsets,
                    *[(index, index + 1) for index in range(len(body))],
                    (len(body), len(text)),
                ],
            }

            if return_offsets_mapping:
                return encoded

            return {"input_ids": encoded["input_ids"]}

        encoded = {
            "input_ids": [*prefix_ids, *[ord(character) for character in text]],
            "offset_mapping": [*prefix_offsets, *[(index, index + 1) for index in range(len(text))]],
        }

        if return_offsets_mapping:
            return encoded

        return {"input_ids": encoded["input_ids"]}


class BoundaryAwareTransformersModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.probe = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> FakeTransformersOutput:
        # ---------------------------------------------------------
        # Prefer the prompt-attached A token, but prefer standalone B
        # if the scorer incorrectly tokenizes the label alone.
        # ---------------------------------------------------------
        del attention_mask
        logits = torch.zeros((*input_ids.shape, 128), dtype=torch.float32)
        logits[:, :, ord("A")] = 6.0
        logits[:, :, ord("B") + 20] = 6.0
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
            offset_rows=[
                [(0, 0), (0, 1), (1, 2), (2, 3)],
                [(0, 0), (0, 1), (1, 2)],
            ],
            prompt_text_len=1,
            max_len=4,
        )

        self.assertEqual(labels, [[-100, 20, 30, -100], [-100, 21, -100, -100]])

    def test_build_continuation_labels_scores_boundary_merged_token(self) -> None:
        # ---------------------------------------------------------
        # Score a token that starts inside the prompt but reaches
        # the continuation text.
        # ---------------------------------------------------------
        labels = build_continuation_labels(
            token_ids=[1, 10, 90],
            offsets=[(0, 0), (0, 1), (1, 3)],
            prompt_text_len=2,
            max_len=2,
            ignored_token_id=-100,
        )

        self.assertEqual(labels, [-100, 90])

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

    def test_transformers_choice_scorer_tokenizes_full_continuation_text(self) -> None:
        # ---------------------------------------------------------
        # Score the token ids from prompt plus continuation, not
        # from the continuation encoded by itself.
        # ---------------------------------------------------------
        scorer = TransformersChoiceScorer(
            model=BoundaryAwareTransformersModel(),
            tokenizer=BoundaryAwareTransformersTokenizer(),
            device=torch.device("cpu"),
            model_source="fake/hf",
            torch_dtype_name="auto",
        )

        losses = scorer.score_continuations(prompt="Question: test\nAnswer:", continuations=("A", "B"))

        self.assertLess(losses[0], losses[1])

    def test_transformers_choice_scorer_scores_boundary_merged_token(self) -> None:
        # ---------------------------------------------------------
        # Keep a merged prompt-answer boundary token visible to the
        # loss instead of masking it as prompt text.
        # ---------------------------------------------------------
        scorer = TransformersChoiceScorer(
            model=FakeTransformersModel(),
            tokenizer=BoundaryMergingTransformersTokenizer(),
            device=torch.device("cpu"),
            model_source="fake/hf",
            torch_dtype_name="auto",
        )

        losses = scorer.score_continuations(prompt="Question:", continuations=("A",))

        self.assertGreater(losses[0], 0.0)

    def test_native_choice_scorer_scores_boundary_merged_token(self) -> None:
        # ---------------------------------------------------------
        # Keep a merged prompt-answer boundary token visible in the
        # native scoring path too.
        # ---------------------------------------------------------
        model = FakeModel()
        tokenizer = BoundaryMergingTokenizer()

        score_native_answer_label(
            model=model,
            tokenizer=tokenizer,
            prompt="Question:",
            answer_label="A",
            device=torch.device("cpu"),
            pad_token_id=0,
            bos_token_id=1,
            max_seq_len=128,
        )

        self.assertEqual(model.labels[-1][-1], 90)

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

    def test_save_result_writes_config_and_csv(self) -> None:
        # ---------------------------------------------------------
        # Save JSON config and per-example CSV rows to the
        # requested output directory.
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
            output_dir = Path(temp_dir) / "nested" / "result"
            with patch("src.eval.jmmlu.runtime.console.print"):
                save_result(
                    result=result,
                    output_dir=output_dir,
                    limit=None,
                    subjects=None,
                )

            config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
            with (output_dir / "result.csv").open(encoding="utf-8", newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual(config["dataset"], "nlp-waseda/JMMLU")
        self.assertEqual(config["overall"]["accuracy"], 1.0)
        self.assertEqual(rows[0]["subject"], "s1")
        self.assertEqual(rows[0]["question"], "q1")
        self.assertEqual(rows[0]["answer"], "A")
        self.assertEqual(rows[0]["prediction"], "A")
        self.assertEqual(rows[0]["correct"], "True")
        self.assertEqual(rows[0]["loss_A"], "0.0")


if __name__ == "__main__":
    unittest.main()
