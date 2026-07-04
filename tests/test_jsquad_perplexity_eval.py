import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn

from src.eval.jsquad_perplexity.dataset import DEFAULT_CONTEXT_PATH
from src.eval.jsquad_perplexity.dataset import JSQUAD_DATASET_ID
from src.eval.jsquad_perplexity.dataset import JSQuADContext
from src.eval.jsquad_perplexity.dataset import load_contexts
from src.eval.jsquad_perplexity.runtime import evaluate_contexts
from src.eval.jsquad_perplexity.runtime import save_result
from src.eval.shared.models import NativeChoiceScorer
from src.eval.shared.models import TextScore
from src.eval.shared.models import TransformersChoiceScorer


class FakeTextTokenizer:
    pad_token = "|<pad>|"
    bos_token = "|<bos>|"

    def token_to_id(self, token: str) -> int:
        # ---------------------------------------------------------
        # Return stable special-token ids for native scorer tests.
        # ---------------------------------------------------------
        token_ids = {
            self.pad_token: 0,
            self.bos_token: 1,
        }
        return token_ids[token]

    def tokenize(self, sentence: str) -> list[int]:
        # ---------------------------------------------------------
        # Encode characters into small ids so text scoring can count
        # labels without depending on a real tokenizer.
        # ---------------------------------------------------------
        return [index + 2 for index, _ in enumerate(sentence)]


class FakeNativeTextModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # ---------------------------------------------------------
        # Keep a parameter for normal torch module behavior and
        # record every text scoring chunk.
        # ---------------------------------------------------------
        self.probe = nn.Parameter(torch.zeros(1))
        self.labels: list[list[int]] = []

    def compute_chunked_loss(self, input_tokens: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # ---------------------------------------------------------
        # Return a fixed mean loss so score_text can be checked as a
        # token-weighted summed loss.
        # ---------------------------------------------------------
        del input_tokens
        self.labels.append([int(token_id) for token_id in labels[0].tolist()])
        return torch.tensor(2.0)


class FakeTransformersTextTokenizer:
    pad_token_id = 0
    eos_token_id = 7
    model_max_length = 2

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict[str, list[int]]:
        # ---------------------------------------------------------
        # Encode text into a short causal LM sequence with one BOS
        # token and deterministic content token ids.
        # ---------------------------------------------------------
        prefix = [1] if add_special_tokens else []
        return {"input_ids": [*prefix, *[index + 2 for index, _ in enumerate(text)]]}


class FakeTransformersTextOutput:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class FakeTransformersTextModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # ---------------------------------------------------------
        # Keep model metadata compatible with HF context length
        # resolution used by the shared scorer.
        # ---------------------------------------------------------
        self.probe = nn.Parameter(torch.zeros(1))
        self.config = type("Config", (), {"max_position_embeddings": 4})()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> FakeTransformersTextOutput:
        # ---------------------------------------------------------
        # Produce uniform logits so each token loss is log(vocab).
        # ---------------------------------------------------------
        del attention_mask
        logits = torch.zeros((*input_ids.shape, 8), dtype=torch.float32, device=input_ids.device)
        return FakeTransformersTextOutput(logits=logits)


class FakePerplexityScorer:
    backend = "fake"
    model_source = "models/test"
    device_name = "cpu"
    torch_dtype_name = "auto"

    def score_text(self, text: str) -> TextScore:
        # ---------------------------------------------------------
        # Return deterministic token-weighted losses based on text
        # length so aggregation can be tested without a model.
        # ---------------------------------------------------------
        token_count = len(text)
        return TextScore(
            loss_sum=float(token_count),
            token_count=token_count,
        )


class JSQuADPerplexityEvalTest(unittest.TestCase):
    def test_saved_contexts_are_unique(self) -> None:
        # ---------------------------------------------------------
        # Verify the generated JSQuAD validation context file is
        # already deduplicated before evaluation starts.
        # ---------------------------------------------------------
        contexts = load_contexts(context_path=DEFAULT_CONTEXT_PATH)
        context_texts = [context.context for context in contexts]

        self.assertEqual(len(contexts), 1145)
        self.assertEqual(len(context_texts), len(set(context_texts)))
        self.assertTrue(contexts[0].context.startswith("梅雨（つゆ、ばいう）は"))

    def test_load_contexts_reads_saved_json(self) -> None:
        # ---------------------------------------------------------
        # Read the compact stored JSON shape used by the JSQuAD
        # perplexity evaluator.
        # ---------------------------------------------------------
        with tempfile.TemporaryDirectory() as temp_dir:
            context_path = Path(temp_dir) / "contexts.json"
            context_path.write_text(
                json.dumps([{"context_id": 7, "context": "本文"}], ensure_ascii=False),
                encoding="utf-8",
            )

            contexts = load_contexts(context_path=context_path)

        self.assertEqual(contexts, [JSQuADContext(context_id=7, context="本文")])

    def test_native_score_text_returns_token_weighted_loss(self) -> None:
        # ---------------------------------------------------------
        # Score native text in multiple chunks and keep total loss
        # weighted by the number of target tokens.
        # ---------------------------------------------------------
        model = FakeNativeTextModel()
        scorer = NativeChoiceScorer(
            model=model,
            tokenizer=FakeTextTokenizer(),
            max_seq_len=1,
            pad_token_id=0,
            bos_token_id=1,
            device=torch.device("cpu"),
            model_source="models/test",
            torch_dtype_name="auto",
        )

        score = scorer.score_text(text="ab")

        self.assertEqual(score.token_count, 2)
        self.assertEqual(score.loss_sum, 4.0)
        self.assertEqual(model.labels, [[2], [3]])

    def test_transformers_score_text_returns_token_weighted_loss(self) -> None:
        # ---------------------------------------------------------
        # Score Transformers text and sum cross-entropy across all
        # visible next-token labels.
        # ---------------------------------------------------------
        scorer = TransformersChoiceScorer(
            model=FakeTransformersTextModel(),
            tokenizer=FakeTransformersTextTokenizer(),
            device=torch.device("cpu"),
            model_source="fake/hf",
            torch_dtype_name="auto",
        )

        score = scorer.score_text(text="ab")

        self.assertEqual(score.token_count, 2)
        self.assertAlmostEqual(score.loss_sum, 2.0 * math.log(8), places=5)

    def test_evaluate_contexts_aggregates_perplexity(self) -> None:
        # ---------------------------------------------------------
        # Aggregate deterministic fake scores into corpus-level
        # token-weighted perplexity.
        # ---------------------------------------------------------
        result = evaluate_contexts(
            scorer=FakePerplexityScorer(),
            contexts=[
                JSQuADContext(context_id=1, context="aa"),
                JSQuADContext(context_id=2, context="bbb"),
            ],
        )

        self.assertEqual(result.dataset, JSQUAD_DATASET_ID)
        self.assertEqual(result.overall.total, 2)
        self.assertEqual(result.overall.token_count, 5)
        self.assertEqual(result.overall.loss, 1.0)
        self.assertAlmostEqual(result.overall.perplexity, math.e)

    def test_save_result_writes_config_and_csv(self) -> None:
        # ---------------------------------------------------------
        # Save JSON config and per-context CSV rows to the requested
        # output directory.
        # ---------------------------------------------------------
        result = evaluate_contexts(
            scorer=FakePerplexityScorer(),
            contexts=[JSQuADContext(context_id=3, context="本文")],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "nested" / "result"
            with patch("src.eval.jsquad_perplexity.runtime.console.print"):
                save_result(result=result, output_dir=output_dir, limit=None)

            config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
            with (output_dir / "result.csv").open(encoding="utf-8", newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual(config["dataset"], "shunk031/JGLUE")
        self.assertEqual(config["config"], "JSQuAD")
        self.assertEqual(config["overall"]["perplexity"], math.e)
        self.assertEqual(rows[0]["context_id"], "3")
        self.assertEqual(rows[0]["context"], "本文")


if __name__ == "__main__":
    unittest.main()
