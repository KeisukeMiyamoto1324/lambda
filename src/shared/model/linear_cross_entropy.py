import torch.nn as nn


def build_linear_cross_entropy_loss(
    ignore_index: int,
) -> nn.Module:
    # ---------------------------------------------------------
    # Import and build the CUDA-only loss when training first uses
    # it, so inference does not require Liger Kernel.
    # ---------------------------------------------------------
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

    return LigerFusedLinearCrossEntropyLoss(
        ignore_index=ignore_index,
        reduction="mean",
    )
