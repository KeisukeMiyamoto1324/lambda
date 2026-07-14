import argparse
from collections.abc import Iterator
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import Dataset

from src.posttraining.artifacts import save_chat_model
from src.posttraining.chat_template import ChatMessage
from src.posttraining.chat_template import tokenize_chat_messages
from src.posttraining.cli import parse_args
from src.posttraining.dataloaders import build_dataloaders
from src.posttraining.dataset import LambdaChatDataset
from src.posttraining.model_setup import DEFAULT_BASE_MODEL_ID
from src.posttraining.model_setup import download_base_model
from src.posttraining.model_setup import load_base_model
from src.posttraining.trainer import build_trainer
from src.shared.model.transformer import DecoderOnlyTransformer


class FakeTokenizer:
    pad_token = "|<pad>|"
    bos_token = "|<bos>|"
    eos_token = "|<eos>|"
    end_of_turn_token = "|<end_of_turn>|"
    system_token = "|<system>|"
    user_token = "|<user>|"
    assistant_token = "|<assistant>|"

    def get_vocab_size(self) -> int:
        # ---------------------------------------------------------
        # Match the saved test model vocabulary size.
        # ---------------------------------------------------------
        return 12

    def token_to_id(self, token: str) -> int:
        # ---------------------------------------------------------
        # Return stable ids for the special tokens needed by model
        # setup without loading a real tokenizer file.
        # ---------------------------------------------------------
        token_ids = {
            self.pad_token: 0,
            self.bos_token: 1,
            self.eos_token: 2,
            self.end_of_turn_token: 3,
            self.system_token: 4,
            self.user_token: 5,
            self.assistant_token: 6,
        }
        return token_ids[token]

    def tokenize(self, sentence: str) -> list[int]:
        # ---------------------------------------------------------
        # Return fixed content ids so chat masks can be tested without
        # loading a tokenizer artifact.
        # ---------------------------------------------------------
        token_ids = {
            "question": [7, 8],
            "answer": [9, 10],
        }
        return token_ids[sentence]


class FakeDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, size: int) -> None:
        # ---------------------------------------------------------
        # Store a fixed dataset size so dataloader step math can be
        # tested without loading the remote SFT dataset.
        # ---------------------------------------------------------
        self.size = size

    def __len__(self) -> int:
        # ---------------------------------------------------------
        # Return the configured number of examples.
        # ---------------------------------------------------------
        return self.size

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # ---------------------------------------------------------
        # Return one tiny packed sample for DataLoader construction.
        # ---------------------------------------------------------
        del index
        tensor = torch.tensor([1], dtype=torch.long)
        return tensor, tensor, tensor, tensor


class FakeStreamingDataset:
    def __init__(self, samples: list[dict[str, object]]) -> None:
        # ---------------------------------------------------------
        # Keep chat records behind the small subset of Hugging Face
        # streaming methods used by LambdaChatDataset.
        # ---------------------------------------------------------
        self.samples = samples

    def select_columns(self, column_names: list[str]) -> "FakeStreamingDataset":
        # ---------------------------------------------------------
        # Preserve records because tests already provide only the
        # message column consumed by the stream.
        # ---------------------------------------------------------
        del column_names
        return self

    def reshard(self) -> "FakeStreamingDataset":
        # ---------------------------------------------------------
        # Return the same local stream without remote shard changes.
        # ---------------------------------------------------------
        return self

    def shuffle(self, seed: int, buffer_size: int) -> "FakeStreamingDataset":
        # ---------------------------------------------------------
        # Accept stream shuffle settings while keeping test order
        # deterministic and easy to assert.
        # ---------------------------------------------------------
        del seed, buffer_size
        return self

    def __iter__(self) -> Iterator[dict[str, object]]:
        # ---------------------------------------------------------
        # Yield the configured records through the streaming API.
        # ---------------------------------------------------------
        yield from self.samples


class PosttrainingModelSetupTest(unittest.TestCase):
    def test_parse_args_rejects_invalid_runtime_values(self) -> None:
        # ---------------------------------------------------------
        # Reject invalid values before model download, dataset loading,
        # or DataLoader construction can begin.
        # ---------------------------------------------------------
        invalid_cases = [
            ("--max-len", "0"),
            ("--learning-rate", "0"),
            ("--lr-warmup-steps", "-1"),
            ("--lr-warmup-steps", "1024"),
            ("--min-learning-rate-ratio", "1.1"),
            ("--batch-size", "0"),
            ("--gradient-accumulation-steps", "0"),
            ("--max-steps", "0"),
            ("--num-workers", "-1"),
            ("--val-batches", "0"),
            ("--val-check-interval", "0"),
            ("--checkpoint-every-n-steps", "0"),
            ("--metric-log-every-n-steps", "0"),
            ("--loss-chunk-size", "0"),
        ]

        for flag, value in invalid_cases:
            with self.subTest(flag=flag), patch("sys.argv", ["train.py", flag, value]), patch(
                "sys.stderr", io.StringIO()
            ):
                with self.assertRaises(SystemExit):
                    parse_args()

    def test_parse_args_rejects_two_resume_sources(self) -> None:
        # ---------------------------------------------------------
        # Prevent checkpoint restoration and fresh weight loading
        # from being requested for the same training run.
        # ---------------------------------------------------------
        argv = [
            "train.py",
            "--resume-from-checkpoint",
            "last.ckpt",
            "--continue-from-model",
            "model.pth",
        ]

        with patch("sys.argv", argv), patch("sys.stderr", io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args()

    def test_parse_args_accepts_hub_push_with_it_repo(self) -> None:
        # ---------------------------------------------------------
        # Accept automatic upload when the instruction-tuned model
        # repository is available through the environment.
        # ---------------------------------------------------------
        with patch("sys.argv", ["train.py", "--push-to-hub"]), patch.dict(
            "os.environ", {"HF_REPO_IT": "user/lambda-it"}
        ):
            args = parse_args()

        self.assertTrue(args.push_to_hub)

    def test_download_base_model_uses_hub_snapshot(self) -> None:
        # ---------------------------------------------------------
        # Resolve Hub model ids through snapshot_download so training
        # always works from local artifacts after download.
        # ---------------------------------------------------------
        with patch("src.posttraining.model_setup.snapshot_download", return_value="/tmp/model") as mocked_download:
            model_dir = download_base_model(base_model_id="user/model")

        self.assertEqual(model_dir, Path("/tmp/model"))
        mocked_download.assert_called_once_with(repo_id="user/model", repo_type="model")

    def test_optimizer_uses_all_trainable_parameters(self) -> None:
        # ---------------------------------------------------------
        # Keep every parameter trainable so full-model fine tuning
        # sends the whole model into the optimizer.
        # ---------------------------------------------------------
        model = DecoderOnlyTransformer(
            num_tokens=12,
            d_model=8,
            num_layers=4,
            num_heads=2,
            d_ff=16,
            pad_token_id=0,
        )
        optimizer = model.configure_optimizers()

        model_parameter_ids = {id(parameter) for parameter in model.parameters()}
        optimizer_parameter_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }

        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
        self.assertEqual(optimizer_parameter_ids, model_parameter_ids)

    def test_load_base_model_loads_pytorch_weights_and_trains_all_layers(self) -> None:
        # ---------------------------------------------------------
        # Load PyTorch weights into the local Lightning model and
        # return metadata for the downloaded architecture.
        # ---------------------------------------------------------
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            model = DecoderOnlyTransformer(
                num_tokens=12,
                d_model=8,
                num_layers=4,
                num_heads=2,
                d_ff=16,
                pad_token_id=0,
            )
            torch.save(model.state_dict(), model_dir / "model.pth")
            (model_dir / "model_config.json").write_text(
                json.dumps(
                    {
                        "max_len": 16,
                        "d_model": 8,
                        "num_layers": 4,
                        "num_heads": 2,
                        "d_ff": 16,
                        "learning_rate": 5e-5,
                        "pad_token_id": 0,
                        "bos_token_id": 1,
                        "eos_token_id": 2,
                    }
                ),
                encoding="utf-8",
            )

            with patch("src.posttraining.model_setup.resolve_device", return_value=torch.device("cpu")):
                loaded_model, model_config = load_base_model(
                    base_model_dir=model_dir,
                    tokenizer=FakeTokenizer(),
                    learning_rate=5e-5,
                    accelerator="cpu",
                    loss_chunk_size=8,
                    lr_warmup_steps=2,
                    lr_total_steps=10,
                    min_learning_rate=1e-5,
                )

        self.assertEqual(model_config["max_len"], 16)
        self.assertEqual(model_config["num_layers"], 4)
        self.assertEqual(loaded_model.loss_chunk_size, 8)
        self.assertEqual(loaded_model.lr_warmup_steps, 2)
        self.assertEqual(loaded_model.lr_total_steps, 10)
        self.assertEqual(loaded_model.min_learning_rate, 1e-5)
        self.assertTrue(all(parameter.requires_grad for parameter in loaded_model.parameters()))

    def test_save_chat_model_persists_pytorch_metadata(self) -> None:
        # ---------------------------------------------------------
        # Save PyTorch metadata with trainable layer provenance
        # included without extra converted artifacts.
        # ---------------------------------------------------------
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            model = DecoderOnlyTransformer(
                num_tokens=12,
                d_model=8,
                num_layers=4,
                num_heads=2,
                d_ff=16,
                pad_token_id=0,
            )
            args = argparse.Namespace(
                max_len=8,
                learning_rate=5e-5,
                base_model_id=DEFAULT_BASE_MODEL_ID,
                posttraining_steps=9,
                devices="auto",
                device_count=1,
                global_batch_size=16,
                global_effective_batch_size=32,
                batch_size=16,
                gradient_accumulation_steps=2,
                lr_warmup_steps=2,
                min_learning_rate=1e-5,
                min_learning_rate_ratio=0.2,
                loss_chunk_size=8,
                validation_cache_path="validation.pt",
                validation_sample_count=128,
            )
            model_config = {
                "max_len": 16,
                "d_model": 8,
                "num_layers": 4,
                "num_heads": 2,
                "d_ff": 16,
                "learning_rate": 5e-5,
                "pad_token_id": 0,
                "bos_token_id": 1,
                "eos_token_id": 2,
            }

            save_chat_model(
                model=model,
                model_dir=model_dir,
                model_config=model_config,
                args=args,
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=2,
                end_of_turn_token_id=11,
            )

            payload = json.loads((model_dir / "model_config.json").read_text())
            model_path_exists = (model_dir / "model.pth").is_file()

        self.assertEqual(payload["base_model_id"], DEFAULT_BASE_MODEL_ID)
        self.assertEqual(payload["training_max_len"], 8)
        self.assertEqual(payload["trainable_layers"], "all")
        self.assertEqual(payload["posttraining_datasets"], ["KeisukeMiyamoto/lambda-chat:train"])
        self.assertEqual(payload["validation_dataset"], "KeisukeMiyamoto/lambda-chat:validation")
        self.assertEqual(payload["posttraining_steps"], 9)
        self.assertEqual(payload["devices"], "auto")
        self.assertEqual(payload["device_count"], 1)
        self.assertEqual(payload["global_batch_size"], 16)
        self.assertEqual(payload["effective_batch_size"], 32)
        self.assertEqual(payload["global_effective_batch_size"], 32)
        self.assertEqual(payload["gradient_accumulation_steps"], 2)
        self.assertEqual(payload["lr_schedule"], "warmup_cosine")
        self.assertEqual(payload["lr_warmup_steps"], 2)
        self.assertEqual(payload["validation_cache_path"], "validation.pt")
        self.assertEqual(payload["validation_sample_count"], 128)
        self.assertEqual(payload["loss_chunk_size"], 8)
        self.assertTrue(model_path_exists)

    def test_lambda_chat_dataset_maps_messages_to_chat_messages(self) -> None:
        # ---------------------------------------------------------
        # Convert lambda-chat message dictionaries into the roles
        # and content expected by the shared chat template.
        # ---------------------------------------------------------
        sample = {
            "id": "id-1",
            "messages": [
                {"role": "user", "content": "質問"},
                {"role": "assistant", "content": "回答"},
            ],
        }

        stream = FakeStreamingDataset(samples=[sample])

        with patch("src.posttraining.dataset.load_dataset", return_value=stream) as mocked_load:
            with patch(
                "src.posttraining.dataset.build_chat_segment",
                return_value=([1], [2]),
            ) as mocked_build:
                dataset = LambdaChatDataset(
                    tokenizer=FakeTokenizer(),
                    split="train",
                    max_len=8,
                    pad_token_id=0,
                    bos_token_id=1,
                    eos_token_id=2,
                    end_of_turn_token_id=3,
                )
                examples = list(dataset)

        messages = mocked_build.call_args.kwargs["messages"]
        mocked_load.assert_called_once_with(
            path="KeisukeMiyamoto/lambda-chat",
            split="train",
            streaming=True,
        )
        self.assertEqual(len(examples), 1)
        self.assertEqual(messages[0].role, "user")
        self.assertEqual(messages[0].content, "質問")
        self.assertEqual(messages[1].role, "assistant")
        self.assertEqual(messages[1].content, "回答")

    def test_chat_tokenization_returns_unpadded_assistant_targets(self) -> None:
        # ---------------------------------------------------------
        # Keep user tokens masked and leave both streams unpadded so
        # several conversations can share one packed sequence.
        # ---------------------------------------------------------
        example = tokenize_chat_messages(
            tokenizer=FakeTokenizer(),
            messages=[
                ChatMessage(role="user", content="question"),
                ChatMessage(role="assistant", content="answer"),
            ],
            max_len=20,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            end_of_turn_token_id=3,
        )

        self.assertEqual(example.input_ids, [1, 5, 7, 8, 3, 6, 9, 10, 3])
        self.assertEqual(example.labels, [0, 0, 0, 0, 0, 9, 10, 3, 2])

    def test_chat_tokenization_keeps_max_len_truncation(self) -> None:
        # ---------------------------------------------------------
        # Preserve the existing leading-context truncation instead
        # of splitting one long conversation into separate segments.
        # ---------------------------------------------------------
        example = tokenize_chat_messages(
            tokenizer=FakeTokenizer(),
            messages=[ChatMessage(role="assistant", content="answer")],
            max_len=2,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            end_of_turn_token_id=3,
        )

        self.assertEqual(example.input_ids, [1, 6])
        self.assertEqual(example.labels, [0, 9])

    def test_lambda_chat_dataset_bucket_packs_conversations(self) -> None:
        # ---------------------------------------------------------
        # Pack short conversations by best fit while resetting
        # positions and isolating each conversation by segment id.
        # ---------------------------------------------------------
        samples = [
            {"messages": [{"role": "user", "content": str(index)}]}
            for index in range(3)
        ]
        segments = [
            ([1, 10, 11, 12], [0, 10, 11, 12]),
            ([1, 20], [0, 20]),
            ([1, 30, 31], [0, 30, 31]),
        ]

        stream = FakeStreamingDataset(samples=samples)

        with patch("src.posttraining.dataset.load_dataset", return_value=stream):
            with patch("src.posttraining.dataset.build_chat_segment", side_effect=segments):
                dataset = LambdaChatDataset(
                    tokenizer=FakeTokenizer(),
                    split="validation",
                    max_len=6,
                    pad_token_id=0,
                    bos_token_id=1,
                    eos_token_id=2,
                    end_of_turn_token_id=3,
                )
                examples = list(dataset)

        input_ids, labels, position_ids, segment_ids = examples[0]
        remaining_input_ids, _, remaining_position_ids, remaining_segment_ids = examples[1]
        self.assertEqual(len(examples), 2)
        self.assertEqual(input_ids.tolist(), [1, 10, 11, 12, 1, 20])
        self.assertEqual(labels.tolist(), [0, 10, 11, 12, 0, 20])
        self.assertEqual(position_ids.tolist(), [0, 1, 2, 3, 0, 1])
        self.assertEqual(segment_ids.tolist(), [0, 0, 0, 0, 1, 1])
        self.assertEqual(remaining_input_ids.tolist(), [1, 30, 31, 0, 0, 0])
        self.assertEqual(remaining_position_ids.tolist(), [0, 1, 2, 0, 0, 0])
        self.assertEqual(remaining_segment_ids.tolist(), [0, 0, 0, -1, -1, -1])

    def test_build_dataloaders_uses_stream_and_validation_cache(self) -> None:
        # ---------------------------------------------------------
        # Keep training iterable while loading fixed validation
        # tensors from an existing local cache.
        # ---------------------------------------------------------
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "validation.pt"
            cache_path.touch()
            fake_datasets = [FakeDataset(size=5), FakeDataset(size=2)]

            with patch(
                "src.posttraining.dataloaders.LambdaChatDataset",
                side_effect=fake_datasets,
            ) as mocked_stream, patch(
                "src.posttraining.dataloaders.LocalTokenizedDataset",
                return_value=FakeDataset(size=2),
            ) as mocked_cache:
                train_dataloader, validation_dataloader = build_dataloaders(
                    tokenizer=FakeTokenizer(),
                    max_len=8,
                    batch_size=2,
                    num_workers=0,
                    accelerator="cpu",
                    validation_cache_path=cache_path,
                    validation_sample_count=2,
                    shuffle_seed=17,
                )

        self.assertEqual(len(train_dataloader), 3)
        self.assertEqual(len(validation_dataloader), 1)
        self.assertTrue(mocked_stream.call_args_list[0].kwargs["repeat_forever"])
        self.assertEqual(mocked_cache.call_args.kwargs["path"], cache_path)

    def test_build_trainer_validates_by_global_step(self) -> None:
        # ---------------------------------------------------------
        # Allow validation intervals larger than one epoch by using
        # Lightning's global-step validation mode.
        # ---------------------------------------------------------
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = build_trainer(
                model_dir=Path(temp_dir),
                stage_name="lambda-chat",
                max_steps=723,
                accelerator="cpu",
                precision="32-true",
                val_check_interval=500,
                val_batches=8,
                checkpoint_every_n_steps=1000,
                metric_log_every_n_steps=50,
                gradient_accumulation_steps=2,
            )

        self.assertIsNone(trainer.check_val_every_n_epoch)
        self.assertEqual(trainer.val_check_interval, 500)
        self.assertEqual(trainer.accumulate_grad_batches, 2)


if __name__ == "__main__":
    unittest.main()
