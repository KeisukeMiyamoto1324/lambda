import torch.nn as nn
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss


def build_linear_cross_entropy_loss(
    ignore_index: int,
) -> nn.Module:
    # ---------------------------------------------------------
    # Build the CUDA fused projection and cross entropy loss.
    # ---------------------------------------------------------
    return LigerFusedLinearCrossEntropyLoss(
        ignore_index=ignore_index,
        reduction="mean",
    )
