from typing import Any


def pad_token_row(token_ids: list[int], max_len: int, pad_token_id: int) -> list[int]:
    # ---------------------------------------------------------
    # Right-pad one token row for batched Transformers scoring.
    # ---------------------------------------------------------
    return [*token_ids, *[pad_token_id for _ in range(max_len - len(token_ids))]]


def encode_hf_text(tokenizer: Any, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    # ---------------------------------------------------------
    # Keep token ids and source text ranges from the same full
    # encoding so boundary merges are handled correctly.
    # ---------------------------------------------------------
    encoded = tokenizer(text, add_special_tokens=True, return_offsets_mapping=True)
    token_ids = [int(token_id) for token_id in encoded["input_ids"]]
    offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
    return token_ids, offsets


def build_continuation_labels(
    token_ids: list[int],
    offsets: list[tuple[int, int]],
    prompt_text_len: int,
    max_len: int,
    ignored_token_id: int,
) -> list[int]:
    # ---------------------------------------------------------
    # Score tokens whose source text reaches the continuation.
    # This also scores tokens merged across the prompt boundary.
    # ---------------------------------------------------------
    labels = [ignored_token_id for _ in range(max_len)]

    for full_index in range(1, len(token_ids)):
        if offsets[full_index][1] > prompt_text_len:
            labels[full_index - 1] = token_ids[full_index]

    return labels


def build_hf_labels(
    full_token_ids: list[list[int]],
    offset_rows: list[list[tuple[int, int]]],
    prompt_text_len: int,
    max_len: int,
) -> list[list[int]]:
    # ---------------------------------------------------------
    # Build labels aligned for causal LM shifting. Prompt-only
    # and pad positions are ignored with -100.
    # ---------------------------------------------------------
    return [
        build_continuation_labels(
            token_ids=token_ids,
            offsets=offsets,
            prompt_text_len=prompt_text_len,
            max_len=max_len,
            ignored_token_id=-100,
        )
        for token_ids, offsets in zip(full_token_ids, offset_rows, strict=True)
    ]
