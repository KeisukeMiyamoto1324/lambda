import torch

from src.eval.jmmlu.dataset import ANSWER_LABELS
from src.eval.jmmlu.dataset import JmmluExample
from src.shared.tokenizer import ByteLevelBPE


def build_prompt(example: JmmluExample) -> str:
    # ---------------------------------------------------------
    # Build the common MMLU-style zero-shot prompt with four
    # labeled choices and an answer label target.
    # ---------------------------------------------------------
    choices_text = "\n".join(
        f"{label}. {choice}"
        for label, choice in zip(ANSWER_LABELS, example.choices, strict=True)
    )
    return f"Question: {example.question}\n{choices_text}\nAnswer:"


def score_answer_label(
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
    # Mask all prompt tokens and compute autoregressive loss only
    # for the candidate answer label tokens.
    # ---------------------------------------------------------
    prompt_token_ids = [bos_token_id, *tokenizer.tokenize(prompt)]
    answer_token_ids = tokenizer.tokenize(f" {answer_label}")
    full_token_ids = [*prompt_token_ids, *answer_token_ids]
    input_token_ids = full_token_ids[:-1]

    if len(input_token_ids) > max_seq_len:
        raise ValueError(f"JMMLU prompt is longer than model context: {len(input_token_ids)} > {max_seq_len}")

    label_token_ids = [pad_token_id for _ in input_token_ids]
    answer_start_index = len(prompt_token_ids)

    for full_index in range(answer_start_index, len(full_token_ids)):
        label_token_ids[full_index - 1] = full_token_ids[full_index]

    input_tokens = torch.tensor([input_token_ids], dtype=torch.long, device=device)
    labels = torch.tensor([label_token_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        loss = model.compute_chunked_loss(input_tokens=input_tokens, labels=labels)

    return float(loss.item())


def predict_answer(
    model: torch.nn.Module,
    tokenizer: ByteLevelBPE,
    example: JmmluExample,
    device: torch.device,
    pad_token_id: int,
    bos_token_id: int,
    max_seq_len: int,
) -> str:
    # ---------------------------------------------------------
    # Score each answer label and choose the label with the
    # lowest language-model loss.
    # ---------------------------------------------------------
    prompt = build_prompt(example=example)
    losses = [
        score_answer_label(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            answer_label=answer_label,
            device=device,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            max_seq_len=max_seq_len,
        )
        for answer_label in ANSWER_LABELS
    ]
    best_index = min(range(len(losses)), key=lambda index: losses[index])
    return ANSWER_LABELS[best_index]
