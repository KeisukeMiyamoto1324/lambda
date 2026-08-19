from pathlib import Path
import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

import torch

from src.eval.shared.scorer_loader import load_native_choice_scorer
from src.eval.shared.scorer_loader import load_transformers_choice_scorer


class EvalScorerLoaderTest(unittest.TestCase):
    def test_native_scorer_uses_resolved_inference_device(self) -> None:
        # ---------------------------------------------------------
        # Move the native evaluation model to the shared inference
        # device and expose the same device through the scorer.
        # ---------------------------------------------------------
        model = MagicMock()
        model.to.return_value = model
        tokenizer = MagicMock()
        tokenizer.get_vocab_size.return_value = 16
        tokenizer.token_to_id.side_effect = [0, 1]
        device = torch.device("mps")

        with (
            patch.object(Path, "exists", return_value=True),
            patch("src.eval.shared.scorer_loader.ByteLevelBPE.load", return_value=tokenizer),
            patch(
                "src.eval.shared.scorer_loader.load_pytorch_model",
                return_value=(model, {"max_len": 128}),
            ),
            patch("src.eval.shared.scorer_loader.resolve_inference_device", return_value=device),
            patch("src.eval.shared.scorer_loader.build_native_prompt_formatter", return_value=MagicMock()),
        ):
            scorer = load_native_choice_scorer(
                model_source="models/test-model",
                torch_dtype_name="auto",
                prompt_format="base",
            )

        model.to.assert_called_once_with(device=device)
        self.assertEqual(scorer.device, device)

    def test_transformers_scorer_uses_resolved_inference_device(self) -> None:
        # ---------------------------------------------------------
        # Move the Transformers evaluation model to the shared
        # inference device and store it in the scorer.
        # ---------------------------------------------------------
        model = MagicMock()
        model.to.return_value = model
        tokenizer = MagicMock()
        device = torch.device("mps")

        with (
            patch("src.eval.shared.scorer_loader.AutoTokenizer.from_pretrained", return_value=tokenizer),
            patch("src.eval.shared.scorer_loader.AutoModelForCausalLM.from_pretrained", return_value=model),
            patch("src.eval.shared.scorer_loader.resolve_inference_device", return_value=device),
            patch("src.eval.shared.scorer_loader.build_hf_prompt_formatter", return_value=MagicMock()),
        ):
            scorer = load_transformers_choice_scorer(
                model_source="test/model",
                torch_dtype_name="auto",
                trust_remote_code=False,
                prompt_format="base",
            )

        model.to.assert_called_once_with(device=device)
        self.assertEqual(scorer.device, device)


if __name__ == "__main__":
    unittest.main()
