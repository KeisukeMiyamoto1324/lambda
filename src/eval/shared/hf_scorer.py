from dataclasses import dataclass
from typing import Any

import torch

from src.eval.shared.labeling import build_hf_labels
from src.eval.shared.labeling import encode_hf_text
from src.eval.shared.labeling import pad_token_row
from src.eval.shared.losses import compute_loss_sum
from src.eval.shared.losses import compute_row_losses
from src.eval.shared.losses import merge_text_scores
from src.eval.shared.scorer_types import TextScore


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
        encoded_rows = [
            encode_hf_text(tokenizer=self.tokenizer, text=f"{prompt}{continuation}")
            for continuation in continuations
        ]
        full_token_ids = [token_ids for token_ids, _ in encoded_rows]
        offset_rows = [offsets for _, offsets in encoded_rows]
        max_len = max(len(token_ids) for token_ids in full_token_ids)
        pad_token_id = resolve_hf_pad_token_id(tokenizer=self.tokenizer)
        input_rows = [
            pad_token_row(token_ids=token_ids, max_len=max_len, pad_token_id=pad_token_id)
            for token_ids in full_token_ids
        ]
        labels = build_hf_labels(
            full_token_ids=full_token_ids,
            offset_rows=offset_rows,
            prompt_text_len=len(prompt),
            max_len=max_len,
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

    def score_text(self, text: str) -> TextScore:
        # ---------------------------------------------------------
        # Score full text perplexity with Transformers while keeping
        # each forward pass inside the model context window.
        # ---------------------------------------------------------
        return score_hf_text(
            model=self.model,
            tokenizer=self.tokenizer,
            text=text,
            device=self.device,
        )


def score_hf_text(
    model: Any,
    tokenizer: Any,
    text: str,
    device: torch.device,
) -> TextScore:
    # ---------------------------------------------------------
    # Convert text to next-token prediction chunks using the same
    # tokenizer path as regular Transformers causal LM scoring.
    # ---------------------------------------------------------
    token_ids = [int(token_id) for token_id in tokenizer(text, add_special_tokens=True)["input_ids"]]
    max_seq_len = resolve_hf_max_seq_len(model=model, tokenizer=tokenizer)
    input_token_ids = token_ids[:-1]
    label_token_ids = token_ids[1:]
    chunk_starts = range(0, len(input_token_ids), max_seq_len)
    chunk_scores = [
        score_hf_text_chunk(
            model=model,
            input_token_ids=input_token_ids[chunk_start : chunk_start + max_seq_len],
            label_token_ids=label_token_ids[chunk_start : chunk_start + max_seq_len],
            device=device,
        )
        for chunk_start in chunk_starts
    ]
    return merge_text_scores(scores=chunk_scores)


def score_hf_text_chunk(
    model: Any,
    input_token_ids: list[int],
    label_token_ids: list[int],
    device: torch.device,
) -> TextScore:
    # ---------------------------------------------------------
    # Run one Transformers model chunk and sum cross-entropy over
    # all visible next-token labels.
    # ---------------------------------------------------------
    input_ids = torch.tensor([input_token_ids], dtype=torch.long, device=device)
    labels = torch.tensor([label_token_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)

    loss_sum = compute_loss_sum(logits=output.logits, labels=labels)
    return TextScore(
        loss_sum=loss_sum,
        token_count=len(label_token_ids),
    )


def resolve_hf_max_seq_len(model: Any, tokenizer: Any) -> int:
    # ---------------------------------------------------------
    # Pick the smallest real context limit exposed by the model or
    # tokenizer so chunks fit both interfaces.
    # ---------------------------------------------------------
    candidates = [
        int(value)
        for value in (
            getattr(tokenizer, "model_max_length", None),
            getattr(getattr(model, "config", None), "max_position_embeddings", None),
        )
        if isinstance(value, int) and 0 < value < 1_000_000
    ]

    if not candidates:
        raise ValueError("Hugging Face model or tokenizer must define a finite context length")

    return min(candidates)


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
