from dataclasses import dataclass

import torch

from src.eval.shared.labeling import build_continuation_labels
from src.eval.shared.losses import merge_text_scores
from src.eval.shared.scorer_types import TextScore
from src.shared.tokenizer import ByteLevelBPE


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

    def score_text(self, text: str) -> TextScore:
        # ---------------------------------------------------------
        # Score full text perplexity with the native decoder while
        # splitting long inputs across the model context window.
        # ---------------------------------------------------------
        return score_native_text(
            model=self.model,
            tokenizer=self.tokenizer,
            text=text,
            device=self.device,
            bos_token_id=self.bos_token_id,
            max_seq_len=self.max_seq_len,
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
    encoding = tokenizer.tokenizer.encode(f"{prompt}{continuation}")
    full_token_ids = [bos_token_id, *[int(token_id) for token_id in encoding.ids]]
    offset_row = [(0, 0), *[(int(start), int(end)) for start, end in encoding.offsets]]
    input_token_ids = full_token_ids[:-1]

    if len(input_token_ids) > max_seq_len:
        raise ValueError(f"Evaluation prompt is longer than model context: {len(input_token_ids)} > {max_seq_len}")

    label_token_ids = build_continuation_labels(
        token_ids=full_token_ids,
        offsets=offset_row,
        prompt_text_len=len(prompt),
        max_len=len(input_token_ids),
        ignored_token_id=pad_token_id,
    )

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


def score_native_text(
    model: torch.nn.Module,
    tokenizer: ByteLevelBPE,
    text: str,
    device: torch.device,
    bos_token_id: int,
    max_seq_len: int,
) -> TextScore:
    # ---------------------------------------------------------
    # Convert text to next-token prediction chunks. The BOS token
    # is context only and is not counted in perplexity.
    # ---------------------------------------------------------
    token_ids = [bos_token_id, *tokenizer.tokenize(text)]
    input_token_ids = token_ids[:-1]
    label_token_ids = token_ids[1:]
    chunk_starts = range(0, len(input_token_ids), max_seq_len)
    chunk_scores = [
        score_native_text_chunk(
            model=model,
            input_token_ids=input_token_ids[chunk_start : chunk_start + max_seq_len],
            label_token_ids=label_token_ids[chunk_start : chunk_start + max_seq_len],
            device=device,
        )
        for chunk_start in chunk_starts
    ]
    return merge_text_scores(scores=chunk_scores)


def score_native_text_chunk(
    model: torch.nn.Module,
    input_token_ids: list[int],
    label_token_ids: list[int],
    device: torch.device,
) -> TextScore:
    # ---------------------------------------------------------
    # Run one native model chunk and convert mean token loss back
    # to summed loss for corpus-level perplexity.
    # ---------------------------------------------------------
    input_tokens = torch.tensor([input_token_ids], dtype=torch.long, device=device)
    labels = torch.tensor([label_token_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        loss = model.compute_chunked_loss(input_tokens=input_tokens, labels=labels)

    token_count = len(label_token_ids)
    return TextScore(
        loss_sum=float(loss.item()) * token_count,
        token_count=token_count,
    )
