import torch
import torch.nn.functional as F

from src.eval.shared.scorer_types import TextScore


def compute_loss_sum(logits: torch.Tensor, labels: torch.Tensor) -> float:
    # ---------------------------------------------------------
    # Sum cross-entropy for aligned next-token labels. The caller
    # already shifted input ids and labels by one position.
    # ---------------------------------------------------------
    losses = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        reduction="sum",
    )
    return float(losses.item())


def merge_text_scores(scores: list[TextScore]) -> TextScore:
    # ---------------------------------------------------------
    # Combine chunk scores without averaging twice so corpus
    # perplexity is weighted by token count.
    # ---------------------------------------------------------
    token_count = sum(score.token_count for score in scores)

    if token_count == 0:
        raise ValueError("Text must contain at least one scored token")

    return TextScore(
        loss_sum=sum(score.loss_sum for score in scores),
        token_count=token_count,
    )


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
