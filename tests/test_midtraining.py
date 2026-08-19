import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.midtraining.cli import parse_args
from src.midtraining.training_corpus_cases import MIDTRAINING_TRAIN_CORPUS_CASE
from src.midtraining.training_corpus_cases import MIDTRAINING_VALIDATION_CORPUS_CASE


class MidtrainingTest(unittest.TestCase):
    def test_corpora_use_registered_train_and_validation_splits(self) -> None:
        # ---------------------------------------------------------
        # Use the complete registered splits and the synthetic
        # textbook rewrite text for training and validation.
        # ---------------------------------------------------------
        self.assertEqual(
            (
                MIDTRAINING_TRAIN_CORPUS_CASE.dataset_path,
                MIDTRAINING_TRAIN_CORPUS_CASE.config_name,
                MIDTRAINING_TRAIN_CORPUS_CASE.split,
                MIDTRAINING_TRAIN_CORPUS_CASE.text_column,
                MIDTRAINING_VALIDATION_CORPUS_CASE.dataset_path,
                MIDTRAINING_VALIDATION_CORPUS_CASE.config_name,
                MIDTRAINING_VALIDATION_CORPUS_CASE.split,
                MIDTRAINING_VALIDATION_CORPUS_CASE.text_column,
            ),
            (
                "KeisukeMiyamoto/SyntheticTextbook-jp",
                "default",
                "train",
                "rewrite",
                "KeisukeMiyamoto/SyntheticTextbook-jp",
                "default",
                "validation",
                "rewrite",
            ),
        )

    def test_parse_args_requires_pretrained_model_directory(self) -> None:
        # ---------------------------------------------------------
        # Reject a missing source model before any dataset stream or
        # training output is initialized.
        # ---------------------------------------------------------
        with patch("sys.argv", ["train.py", "--model-path", "missing"]):
            with patch("sys.stderr", io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args()

    def test_parse_args_rejects_invalid_step_budget(self) -> None:
        # ---------------------------------------------------------
        # Reject non-positive step values before creating datasets
        # or loading the pretrained model weights.
        # ---------------------------------------------------------
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)

            for file_name in ["model.pth", "model_config.json", "tokenizer.json"]:
                (model_dir / file_name).touch()

            argv = [
                "train.py",
                "--model-path",
                str(model_dir),
                "--max-steps",
                "0",
            ]

            with patch("sys.argv", argv), patch("sys.stderr", io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args()

    def test_parse_args_rejects_invalid_runtime_values(self) -> None:
        # ---------------------------------------------------------
        # Reject values that would otherwise fail later in dataset
        # packing, DataLoader setup, or training computation.
        # ---------------------------------------------------------
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)

            for file_name in ["model.pth", "model_config.json", "tokenizer.json"]:
                (model_dir / file_name).touch()

            invalid_cases = [
                ("--max-len", "0"),
                ("--batch-size", "0"),
                ("--val-batches", "0"),
                ("--val-check-interval", "0"),
                ("--checkpoint-every-n-steps", "0"),
                ("--metric-log-every-n-steps", "0"),
            ]

            for flag, value in invalid_cases:
                argv = [
                    "train.py",
                    "--model-path",
                    str(model_dir),
                    flag,
                    value,
                ]

                with self.subTest(flag=flag), patch("sys.argv", argv), patch("sys.stderr", io.StringIO()):
                    with self.assertRaises(SystemExit):
                        parse_args()

    def test_parse_args_rejects_removed_validation_split_options(self) -> None:
        # ---------------------------------------------------------
        # Reject obsolete hash split settings because midtraining
        # now uses registered train and validation dataset splits.
        # ---------------------------------------------------------
        removed_options = ["--val-split-modulo", "--val-split-index"]

        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)

            for file_name in ["model.pth", "model_config.json", "tokenizer.json"]:
                (model_dir / file_name).touch()

            for option in removed_options:
                argv = ["train.py", "--model-path", str(model_dir), option, "1"]

                with self.subTest(option=option), patch("sys.argv", argv), patch(
                    "sys.stderr",
                    io.StringIO(),
                ):
                    with self.assertRaises(SystemExit):
                        parse_args()

    def test_parse_args_rejects_invalid_lr_schedule(self) -> None:
        # ---------------------------------------------------------
        # Reject schedule values that cannot form a bounded warmup
        # and decay interval before training starts.
        # ---------------------------------------------------------
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)

            for file_name in ["model.pth", "model_config.json", "tokenizer.json"]:
                (model_dir / file_name).touch()

            argv = [
                "train.py",
                "--model-path",
                str(model_dir),
                "--min-learning-rate-ratio",
                "1.1",
            ]

            with patch("sys.argv", argv), patch("sys.stderr", io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args()

    def test_parse_args_accepts_longer_context_than_source_model(self) -> None:
        # ---------------------------------------------------------
        # Allow midtraining to request longer context than the saved
        # pretraining config because the model rebuild handles RoPE.
        # ---------------------------------------------------------
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)

            for file_name in ["model.pth", "model_config.json", "tokenizer.json"]:
                (model_dir / file_name).touch()

            argv = [
                "train.py",
                "--model-path",
                str(model_dir),
                "--max-len",
                "4096",
            ]

            with patch("sys.argv", argv):
                args = parse_args()

        self.assertEqual(args.max_len, 4096)

    def test_parse_args_accepts_hub_push_with_mid_repo(self) -> None:
        # ---------------------------------------------------------
        # Accept automatic upload when the midtraining model
        # repository is available through the environment.
        # ---------------------------------------------------------
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)

            for file_name in ["model.pth", "model_config.json", "tokenizer.json"]:
                (model_dir / file_name).touch()

            argv = ["train.py", "--model-path", str(model_dir), "--push-to-hub"]

            with patch("sys.argv", argv), patch.dict(
                "os.environ",
                {"HF_REPO_MID": "user/lambda-mid"},
            ):
                args = parse_args()

        self.assertTrue(args.push_to_hub)


if __name__ == "__main__":
    unittest.main()
