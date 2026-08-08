import torch
import torch.nn as nn
import torch.nn.functional as F
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss


class TorchLinearCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index: int) -> None:
        super().__init__()
        self.ignore_index = ignore_index

    def forward(
        self,
        weight: torch.Tensor,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        # ---------------------------------------------------------
        # Project hidden states with the shared output parameters and
        # average cross entropy over non-padding target tokens.
        # ---------------------------------------------------------
        logits = F.linear(hidden_states, weight=weight, bias=bias)
        return F.cross_entropy(
            logits,
            labels,
            ignore_index=self.ignore_index,
            reduction="mean",
        )


def build_linear_cross_entropy_loss(
    ignore_index: int,
    use_fused_kernel: bool,
) -> nn.Module:
    # ---------------------------------------------------------
    # Select the explicit training backend once during model setup so
    # every loss invocation follows one stable execution path.
    # ---------------------------------------------------------
    if use_fused_kernel:
        return LigerFusedLinearCrossEntropyLoss(
            ignore_index=ignore_index,
            reduction="mean",
        )

    return TorchLinearCrossEntropyLoss(ignore_index=ignore_index)
