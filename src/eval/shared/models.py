from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Protocol

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from src.inference_base.generation import resolve_torch_dtype
from src.shared.device_utils import resolve_device
from src.shared.pytorch_artifacts import load_pytorch_model
from src.shared.tokenizer import ByteLevelBPE


class ChoiceScorer(Protocol):
    backend: str
    model_source: str
    device_name: str
    torch_dtype_name: str

    def score_continuations(self, prompt: str, continuations: tuple[str, ...]) -> list[float]:
        ...

    def score_answer_labels(self, prompt: str, answer_labels: tuple[str, ...]) -> list[float]:
        ...


@dataclass
class NativeChoiceScorer:
    model: torch.nn.Module
    tokenizer: ByteLevelBPE
    max_seq_len: int
    pad_token_id: int
    bos_token_id: int
    device: torch.device
    model_source: str
    torch_dtype_name: str
    backend: str = "native"

    @property
    def device_name(self) -> str:
        return self.device.type

    def score_continuations(self, prompt: str, continuations: tuple[str, ...]) -> list[float]:
        # ---------------------------------------------------------
        # Score candidate continuations with the native loss path
        # used by this repository's decoder-only Transformer.
        # ---------------------------------------------------------
        return [
            score_native_continuation(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=prompt,
                continuation=continuation,
                device=self.device,
                pad_token_id=self.pad_token_id,
                bos_token_id=self.bos_token_id,
                max_seq_len=self.max_seq_len,
            )
            for continuation in continuations
        ]

    def score_answer_labels(self, prompt: str, answer_labels: tuple[str, ...]) -> list[float]:
        # ---------------------------------------------------------
        # Keep old JMMLU label scoring behavior by adding one space
        # before each answer label.
        # ---------------------------------------------------------
        continuations = tuple(f" {answer_label}" for answer_label in answer_labels)
        return self.score_continuations(prompt=prompt, continuations=continuations)


@dataclass
class TransformersChoiceScorer:
    model: Any
    tokenizer: Any
    device: torch.device
    model_source: str
    torch_dtype_name: str
    backend: str = "hf"

    @property
    def device_name(self) -> str:
        return self.device.type

    def score_continuations(self, prompt: str, continuations: tuple[str, ...]) -> list[float]:
        # ---------------------------------------------------------
        # Tokenize each full prompt plus continuation because some
        # tokenizers produce different suffix ids at the boundary.
        # ---------------------------------------------------------
        encoded_prompt = self.tokenizer(prompt, add_special_tokens=True)
        prompt_ids = [int(token_id) for token_id in encoded_prompt["input_ids"]]
        full_token_ids = [
            [
                int(token_id)
                for token_id in self.tokenizer(
                    f"{prompt}{continuation}",
                    add_special_tokens=True,
                )["input_ids"]
            ]
            for continuation in continuations
        ]
        max_len = max(len(token_ids) for token_ids in full_token_ids)
        pad_token_id = resolve_hf_pad_token_id(tokenizer=self.tokenizer)
        input_rows = [
            pad_token_row(token_ids=token_ids, max_len=max_len, pad_token_id=pad_token_id)
            for token_ids in full_token_ids
        ]
        labels = build_hf_labels(
            full_token_ids=full_token_ids,
            prompt_len=len(prompt_ids),
            max_len=max_len,
            pad_token_id=pad_token_id,
        )

        input_ids = torch.tensor(input_rows, dtype=torch.long, device=self.device)
        label_ids = torch.tensor(labels, dtype=torch.long, device=self.device)
        attention_mask = input_ids.ne(pad_token_id).to(dtype=torch.long)

        with torch.no_grad():
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)

        return compute_row_losses(logits=output.logits, labels=label_ids)

    def score_answer_labels(self, prompt: str, answer_labels: tuple[str, ...]) -> list[float]:
        # ---------------------------------------------------------
        # Keep old JMMLU label scoring behavior by adding one space
        # before each answer label.
        # ---------------------------------------------------------
        continuations = tuple(f" {answer_label}" for answer_label in answer_labels)
        return self.score_continuations(prompt=prompt, continuations=continuations)


def load_choice_scorer(
    model_source: str,
    backend: str,
    torch_dtype_name: str,
    trust_remote_code: bool,
) -> ChoiceScorer:
    # ---------------------------------------------------------
    # Resolve the backend from the model source and return one
    # scorer object with a shared evaluation interface.
    # ---------------------------------------------------------
    resolved_backend = resolve_backend(model_source=model_source, backend=backend)

    if resolved_backend == "native":
        return load_native_choice_scorer(
            model_source=model_source,
            torch_dtype_name=torch_dtype_name,
        )

    return load_transformers_choice_scorer(
        model_source=model_source,
        torch_dtype_name=torch_dtype_name,
        trust_remote_code=trust_remote_code,
    )


def resolve_backend(model_source: str, backend: str) -> str:
    # ---------------------------------------------------------
    # Auto-detect local native artifacts. Hub ids and other model
    # sources use the Transformers backend.
    # ---------------------------------------------------------
    if backend != "auto":
        return backend

    model_path = Path(model_source)

    if model_path.exists() and (model_path / "model.pth").exists() and (model_path / "model_config.json").exists():
        return "native"

    if is_local_model_path(model_source=model_source):
        raise FileNotFoundError(f"Native model artifacts were not found: {model_source}")

    return "hf"


def is_local_model_path(model_source: str) -> bool:
    # ---------------------------------------------------------
    # Treat common path-like model sources as local artifacts so
    # typos do not become confusing Hub requests.
    # ---------------------------------------------------------
    model_path = Path(model_source)
    return model_path.is_absolute() or model_source.startswith(("./", "../", "models/"))


def load_native_choice_scorer(model_source: str, torch_dtype_name: str) -> NativeChoiceScorer:
    # ---------------------------------------------------------
    # Load this project's PyTorch artifacts and tokenizer from a
    # local model directory.
    # ---------------------------------------------------------
    model_dir = Path(model_source)

    if not model_dir.exists():
        raise FileNotFoundError(f"Native model directory does not exist: {model_source}")

    tokenizer = ByteLevelBPE.load(model_dir)
    model, model_config = load_pytorch_model(
        model_dir=model_dir,
        vocab_size=tokenizer.get_vocab_size(),
    )
    device = resolve_device()
    torch_dtype = resolve_torch_dtype(torch_dtype=torch_dtype_name)
    model = model.to(device=device)

    if torch_dtype is not None:
        model = model.to(dtype=torch_dtype)

    model.eval()
    return NativeChoiceScorer(
        model=model,
        tokenizer=tokenizer,
        max_seq_len=int(model_config["max_len"]),
        pad_token_id=tokenizer.token_to_id(tokenizer.pad_token),
        bos_token_id=tokenizer.token_to_id(tokenizer.bos_token),
        device=device,
        model_source=model_source,
        torch_dtype_name=torch_dtype_name,
    )


def load_transformers_choice_scorer(
    model_source: str,
    torch_dtype_name: str,
    trust_remote_code: bool,
) -> TransformersChoiceScorer:
    # ---------------------------------------------------------
    # Load a Hugging Face causal language model with Transformers
    # for external model comparison.
    # ---------------------------------------------------------
    device = resolve_device()
    torch_dtype = resolve_torch_dtype(torch_dtype=torch_dtype_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        trust_remote_code=trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )
    model = model.to(device=device)
    model.eval()
    return TransformersChoiceScorer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        model_source=model_source,
        torch_dtype_name=torch_dtype_name,
    )


def score_native_continuation(
    model: torch.nn.Module,
    tokenizer: ByteLevelBPE,
    prompt: str,
    continuation: str,
    device: torch.device,
    pad_token_id: int,
    bos_token_id: int,
    max_seq_len: int,
) -> float:
    # ---------------------------------------------------------
    # Tokenize the full text so boundary-sensitive tokenizers use
    # the same ids as real generation.
    # ---------------------------------------------------------
    prompt_token_ids = [bos_token_id, *tokenizer.tokenize(prompt)]
    full_token_ids = [bos_token_id, *tokenizer.tokenize(f"{prompt}{continuation}")]
    input_token_ids = full_token_ids[:-1]

    if len(input_token_ids) > max_seq_len:
        raise ValueError(f"Evaluation prompt is longer than model context: {len(input_token_ids)} > {max_seq_len}")

    label_token_ids = [pad_token_id for _ in input_token_ids]
    continuation_start_index = len(prompt_token_ids)

    for full_index in range(continuation_start_index, len(full_token_ids)):
        label_token_ids[full_index - 1] = full_token_ids[full_index]

    input_tokens = torch.tensor([input_token_ids], dtype=torch.long, device=device)
    labels = torch.tensor([label_token_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        loss = model.compute_chunked_loss(input_tokens=input_tokens, labels=labels)

    return float(loss.item())


def score_native_answer_label(
    model: torch.nn.Module,
    tokenizer: ByteLevelBPE,
    prompt: str,
    answer_label: str,
    device: torch.device,
    pad_token_id: int,
    bos_token_id: int,
    max_seq_len: int,
) -> float:
    # ---------------------------------------------------------
    # Keep the old answer-label helper for JMMLU compatibility.
    # ---------------------------------------------------------
    return score_native_continuation(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        continuation=f" {answer_label}",
        device=device,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        max_seq_len=max_seq_len,
    )


def resolve_hf_pad_token_id(tokenizer: Any) -> int:
    # ---------------------------------------------------------
    # Use an existing pad token when available. Otherwise use EOS
    # for padding because padding tokens are fully masked.
    # ---------------------------------------------------------
    if tokenizer.pad_token_id is not None:
        return int(tokenizer.pad_token_id)

    if tokenizer.eos_token_id is not None:
        return int(tokenizer.eos_token_id)

    raise ValueError("Tokenizer must define pad_token_id or eos_token_id")


def pad_token_row(token_ids: list[int], max_len: int, pad_token_id: int) -> list[int]:
    # ---------------------------------------------------------
    # Right-pad one token row for batched Transformers scoring.
    # ---------------------------------------------------------
    return [*token_ids, *[pad_token_id for _ in range(max_len - len(token_ids))]]


def build_hf_labels(
    full_token_ids: list[list[int]],
    prompt_len: int,
    max_len: int,
    pad_token_id: int,
) -> list[list[int]]:
    # ---------------------------------------------------------
    # Build labels aligned for causal LM shifting. Prompt and pad
    # positions are ignored with -100.
    # ---------------------------------------------------------
    label_rows: list[list[int]] = []

    for token_ids in full_token_ids:
        padded_token_ids = pad_token_row(token_ids=token_ids, max_len=max_len, pad_token_id=pad_token_id)
        labels = [-100 for _ in padded_token_ids]

        for full_index in range(prompt_len, len(token_ids)):
            labels[full_index - 1] = token_ids[full_index]

        label_rows.append(labels)

    return label_rows


def compute_row_losses(logits: torch.Tensor, labels: torch.Tensor) -> list[float]:
    # ---------------------------------------------------------
    # Compute mean cross-entropy for each row independently while
    # ignoring masked labels.
    # ---------------------------------------------------------
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, :-1].contiguous()
    losses = F.cross_entropy(
        shifted_logits.view(-1, shifted_logits.size(-1)),
        shifted_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    )
    row_losses = losses.view(shifted_labels.size())
    label_counts = shifted_labels.ne(-100).sum(dim=1).clamp_min(1)
    return [
        float((row_losses[index].sum() / label_counts[index]).item())
        for index in range(shifted_labels.size(dim=0))
    ]
